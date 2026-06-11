import os
import json
import base64
import operator
import boto3
from typing import Annotated, List, TypedDict, Literal, Dict
from PIL import Image
from io import BytesIO

from langgraph.graph import StateGraph, END
from langchain_aws import ChatBedrock


# --- 1. STATE DEFINITION ---
class AgentState(TypedDict):
    # The current user question (can be updated by the Rewriter)
    question: str
    
    # The deconstructed queries from the Analyzer
    queries: List[str]
    
    # Evidence gathered from ChromaDB
    # Reducers allow nodes to APPEND to these lists instead of overwriting them.
    # 'operator.add' ensures that if we search multiple times, we keep ALL findings
    documents: Annotated[List[str], operator.add] 
    images: Annotated[List[str], operator.add]          # Local file paths to PNGs
    audio_transcripts: Annotated[List[str], operator.add] # Text segments from audio
    
    # The synthesized answer from Claude 3.5 / Nova Pro
    generation: str
    
    # Control Flow Variables
    loop_step: int      # Tracks how many times we've tried to find the answer
    is_relevant: str    # "yes", "no", or "maybe" (from Grader)
    is_grounded: str    # "grounded" or "hallucination" (from Grounding Check)


# --- 2. THE MULTIMODAL AGENT CLASS ---
class TeslaMultimodalAgent:
    def __init__(self, chroma_collection):
        self.bedrock = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        self.collection = chroma_collection
        
        # Models
        self.lite_model = "amazon.nova-lite-v1:0" # Fast/Cheap for Logic
        self.pro_model = "arn:aws:bedrock:us-east-1:701879548597:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0" # High Reasoning

    def _invoke_nova(self, prompt):
        body = json.dumps({
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"max_new_tokens": 500, "temperature": 0}
        })
        response = self.bedrock.invoke_model(modelId=self.lite_model, body=body)
        return json.loads(response['body'].read())['output']['message']['content'][0]['text']

    # NODE 1: ANALYZER
    def analyze_query(self, state: AgentState):
        print("---NODE 1: ANALYZING---")
        prompt = f"Deconstruct this Tesla Q4 request: '{state['question']}'. Output JSON with keys 'text_q', 'image_q', 'audio_q'."
        res = self._invoke_nova(prompt)
        try:
            q_json = json.loads(res)
            return {"queries": [q_json['text_q'], q_json['image_q'], q_json['audio_q']], "loop_step": 1}
        except:
            return {"queries": [state['question']], "loop_step": 1}

    # Helper to get the 1024-dim vector from Bedrock
    def _get_embedding(self, text):
        # The exact schema required by Nova Multimodal Embeddings
        body = json.dumps({
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "text": {
                    "value": text,           # Nova uses 'value' for inline strings
                    "truncationMode": "END"  # Required parameter
                },
                "embeddingPurpose": "GENERIC_RETRIEVAL", # Recommended for search
                "embeddingDimension": 1024               # Match your collection
            }
        })
        response = self.bedrock.invoke_model(
            modelId="amazon.nova-2-multimodal-embeddings-v1:0", # Use the same model as ingestion
            body=body
        )
        return json.loads(response['body'].read())['embeddings'][0]['embedding']
    
    # NODE 2: RETRIEVER
    def retrieve_all(self, state: AgentState):
        print(f"---NODE 2: RETRIEVING (Step {state['loop_step']})---")
        docs, imgs, audio = [], [], []

        for q in state["queries"]:
            # Convert text query to 1024-dim vector first before querying ChromaDB, since our collection is indexed with Nova's multimodal embeddings
            # Query unified vector space

            # Ensure q is a string
            if not isinstance(q, str):
                continue
            query_vector = self._get_embedding(q)

            # Query using 'query_embeddings' instead of 'query_texts'
            results = self.collection.query(query_embeddings=[query_vector], n_results=2)

            for i, meta in enumerate(results['metadatas'][0]):
                if meta['type'] == 'text': docs.append(results['documents'][0][i])
                elif meta['type'] == 'image': imgs.append(meta['path'])
                elif meta['type'] == 'audio': audio.append(f"Transcript: {results['documents'][0][i]}")
        return {"documents": docs, "images": list(set(imgs)), "audio_transcripts": audio}

    # NODE 3: GRADER
    def grade_documents(self, state: AgentState):
        print("---NODE 3: GRADING---")
        context = " ".join(state['documents'] + state['audio_transcripts'])
        prompt = f"Does this context contain data to answer '{state['question']}'? Respond ONLY with 'yes' or 'no'."
        grade = self._invoke_nova(prompt).lower()
        return {"is_relevant": "yes" if "yes" in grade else "no"}

    # NODE 5: REWRITER (Loop-back)
    def rewrite_query(self, state: AgentState):
        print("---NODE 5: REWRITING---")
        prompt = f"The search for '{state['question']}' failed. Provide a better, more technical financial search query."
        new_q = self._invoke_nova(prompt)
        return {"question": new_q, "loop_step": state['loop_step'] + 1}

    # NODE 4: GENERATOR (Claude 3.5 Sonnet)
    def generate_answer(self, state: AgentState):
        print("---NODE 4: SYNTHESIZING---")
        
        # Prepare Multimodal Message for Claude
        content = [{"type": "text", "text": f"Question: {state['question']}\nContext: {state['documents']}\nAudio: {state['audio_transcripts']}"}]
        
        # Add local images as base64
        for img_path in state['images'][:3]: # Limit to top 3 images
            with open(img_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image", 
                    "source": {"type": "base64", "media_type": "image/png", "data": b64_img}
                })

        # Final Synthesis
        msg = {"role": "user", "content": content}
        body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 1000, "messages": [msg]})
        response = self.bedrock.invoke_model(modelId=self.pro_model, body=body)
        gen = json.loads(response['body'].read())['content'][0]['text']
        return {"generation": gen}

    # NODE 6: GROUNDING
    def check_grounding(self, state: AgentState):
        print("---NODE 6: GROUNDING CHECK---")
        prompt = f"Fact Check: Is this answer '{state['generation']}' fully supported by '{state['documents']}'? Answer 'grounded' or 'hallucination'."
        res = self._invoke_nova(prompt).lower()
        return {"is_grounded": "grounded" if "grounded" in res else "hallucination"}

# --- 3. GRAPH ORCHESTRATION ---
def build_graph(agent):
    workflow = StateGraph(AgentState)
    
    workflow.add_node("analyzer", agent.analyze_query)
    workflow.add_node("retriever", agent.retrieve_all)
    workflow.add_node("grader", agent.grade_documents)
    workflow.add_node("generator", agent.generate_answer)
    workflow.add_node("rewriter", agent.rewrite_query)
    workflow.add_node("grounding", agent.check_grounding)

    workflow.set_entry_point("analyzer")
    workflow.add_edge("analyzer", "retriever")
    workflow.add_edge("retriever", "grader")

    # Routing Logic
    workflow.add_conditional_edges(
        "grader",
        lambda x: x["is_relevant"],
        {"yes": "generator", "no": "rewriter"}
    )
    workflow.add_edge("rewriter", "retriever") # Loop back to search
    workflow.add_edge("generator", "grounding")
    workflow.add_edge("grounding", END)

    return workflow.compile()

# --- 4. EXECUTION ---
if __name__ == "__main__":
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    coll = client.get_collection("tesla_multimodal_rag")
    
    tesla_agent = TeslaMultimodalAgent(coll)
    app = build_graph(tesla_agent)
    
    final_state = app.invoke({"question": "What was Tesla's profit and what did Elon say about it?", "documents": [], "images": [], "audio_transcripts": []})
    print("\n\nFINAL RESPONSE:\n", final_state["generation"])