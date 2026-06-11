import os
import fitz  # PyMuPDF
import boto3
import json
import base64
import chromadb
from PIL import Image
from io import BytesIO
from langchain_experimental.text_splitter import SemanticChunker
from langchain_aws import BedrockEmbeddings
from moviepy import VideoFileClip, AudioFileClip

# This script handles the PDF (Tesla Q4 Report), extracts text semantically, saves images locally,
# and pushes everything to ChromaDB.


# 1. SETUP CLIENTS
print("\n--- [INFO] Initializing Bedrock Runtime Client ---")
bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )

# Use Nova for embeddings (Unified space for text/images)
# Dimension: 1024 (Standard) or 3072 (High Res)
EMBED_MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="tesla_multimodal_rag")

# 2. SEMANTIC CHUNKING FOR PDF TEXT
def process_pdf_text(pdf_path):
    print("\n--- [INFO] Processing PDF Text ---")
    loader = fitz.open(pdf_path)
    full_text = ""
    for page in loader:
        full_text += page.get_text()
    
    # Semantic Chunker uses Bedrock to find natural topic breaks
    embed_model = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name=os.getenv("AWS_REGION", "us-east-1"))
    chunker = SemanticChunker(embed_model)
    chunks = chunker.create_documents([full_text])
    print(f"Created {len(chunks)} semantic chunks from PDF text.")
    return [c.page_content for c in chunks]

# 3. IMAGE EXTRACTION & EMBEDDING
def process_pdf_images(pdf_path, output_dir="multimodal_rag/data/processed_images"):
    print("\n--- [INFO] Processing PDF Images ---")
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_data = []

    for i in range(len(doc)):
        # get_page_images() : Retrieve a list of images used on a page.
        for img_index, img in enumerate(doc.get_page_images(i)):    
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            ext = base_image["ext"] # This will be 'jpeg', 'png', etc.
            
            # Save locally for the Agent to "see" later
            img_filename = f"page_{i}_img_{img_index}.png"
            img_path = os.path.join(output_dir, img_filename)
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            
            # Get Multimodal Embedding from Nova
            image_b64 = base64.b64encode(img_bytes).decode('utf-8')
            embedding = get_nova_embedding(image_b64=image_b64, image_format=ext)
            
            image_data.append({
                "id": f"img_{i}_{img_index}",
                "vector": embedding,
                "metadata": {"path": img_path, "type": "image", "source": "pdf"}
            })
    return image_data

def get_nova_embedding(text=None, image_b64=None, image_format="png"):
    print("\n--- [INFO] Getting Nova Multimodal Embedding ---")
    # Base structure for Nova Multimodal Embeddings
    body = {
        "schemaVersion": "nova-multimodal-embed-v1",
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX", # Required for RAG indexing
            "embeddingDimension": 1024,          # Matches your ChromaDB setup # Standard for Nova
        }
    }

    # embeddingPurpose: "GENERIC_INDEX": Since you are currently in the ingestion phase, this tells Bedrock to optimize the vector for storage in a database. When you later write your query script, you would change this to "TEXT_RETRIEVAL".
    # embeddingPurpose: Required. Use "GENERIC_INDEX" for ingestion and "TEXT_RETRIEVAL" or "IMAGE_RETRIEVAL" for querying.
    # Unified Space: Since you are using the same model for both text chunks and images, they will be stored in the same vector space in ChromaDB, which is exactly what you want for a Multimodal RAG.

    # Text handling: Must be an object with 'value' AND 'truncationMode'
    if text:
        body["singleEmbeddingParams"]["text"] = {
            "value": text,
            "truncationMode": "END"  # Options: "END", "START", or "NONE"                                        
        }
    
    # Image handling: Must be an object with 'format' and 'source'
    if image_b64:
        body["singleEmbeddingParams"]["image"] = {"format": image_format, "source": {"bytes": image_b64}}
    
    response = bedrock_runtime.invoke_model(
        body=json.dumps(body),
        modelId=EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json"
    )
    response_body = json.loads(response['body'].read())
    # Nova returns 'embedding' as a single list for SINGLE_EMBEDDING tasks
    # print(f"Received embedding of length {len(response_body['embedding'])} from Nova.")
    
    return response_body['embeddings'][0]['embedding']

# 4. EXECUTE THE "PUSH"
def run_ingestion():
    pdf_path = "./multimodal_rag/data/raw_docs/Tesla Q4 & Full Year 2024 Update Deck.pdf"     # tsla-20250129-gen
    
    # Push Text
    print("Chunking and Pushing Text...")
    text_chunks = process_pdf_text(pdf_path)
    for i, chunk in enumerate(text_chunks):
        vec = get_nova_embedding(text=chunk)
        print(f"Pushing Text Chunk ID: txt_{i} with embedding length: {len(vec)}")
        collection.add(ids=[f"txt_{i}"], embeddings=[vec], documents=[chunk], metadatas=[{"type": "text"}])
    print(f"Pushed {len(text_chunks)} text chunks to ChromaDB.")

    # Push Images
    print("\n--- [INFO] Extracting and Pushing Images ---")
    img_data = process_pdf_images(pdf_path)
    print(f"Extracted {len(img_data)} images. Pushing to ChromaDB...")

    for img in img_data:
        print(f"Pushing Image ID: {img['id']} with metadata: {img['metadata']}")
        collection.add(ids=[img["id"]], embeddings=[img["vector"]], metadatas=[img["metadata"]])
    
    print(f"Pushed {len(img_data)} images to ChromaDB.")
    print("Ingestion Complete!")


## Helper functions for video/audio data:

def get_multimodal_embedding(input_data, type="text"):
    """Generic function to get Nova embeddings for any type"""
    print(f"\n--- [INFO] Getting Nova Embedding for {type.upper()} ---")
    body = {
        "schemaVersion": "nova-multimodal-embed-v1",
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": 1024
        }
    }
    
    if type == "text":
        body["singleEmbeddingParams"]["text"] = {
            "value": input_data,
            "truncationMode": "END"  # Options: "END", "START", or "NONE"                                        
        }
    elif type == "image":
        # Nova handles image logic similarly to audio/video
        body["singleEmbeddingParams"]["image"] = {
            "format": "png", 
            "source": {"bytes": input_data}
        }
    elif type == "audio":
        # Nova supports: mp3, wav, ogg (Max 30 seconds for sync API)
        # Note: In production 2026, Nova accepts base64 audio/video segments
        body["singleEmbeddingParams"]["audio"] = {"format": "mp3", "source": {"bytes": input_data}}
    elif type == "video":
        # Nova supports: mp4, mov, mkv, webm, etc. (Max 30 seconds for sync API)
        body["singleEmbeddingParams"]["video"] = {"format": "mp4", "source": {"bytes": input_data}, "embeddingMode": "AUDIO_VIDEO_COMBINED"} # REQUIRED for video

    response = bedrock_runtime.invoke_model(
        body=json.dumps(body),
        modelId=EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response['body'].read())
    return response_body['embeddings'][0]['embedding']

# 2. ALIGNED SLICING LOGIC
def ingest_media_segments(file_path, modality="video", segment_duration=30):
    """Slices media and pushes to ChromaDB with temporal metadata"""
    clip = VideoFileClip(file_path) if modality == "video" else AudioFileClip(file_path)
    duration = clip.duration
    
    processed_dir = f"data/processed_{modality}"
    os.makedirs(processed_dir, exist_ok=True)

    ## We will process only the first 5 minutes to save you cost
    limit = min(duration, 300) 
    
    for start_t in range(0, int(limit), segment_duration):
        end_t = min(start_t + segment_duration, duration)
        
        # Create the segment
        subclip = clip.subclip(start_t, end_t)
        segment_filename = f"segment_{start_t}_{end_t}.mp4"
        segment_path = os.path.join(processed_dir, segment_filename)
        
        # Write the small file to disk (Agent will use this for final reasoning)
        if modality == "video":
            subclip.write_videofile(segment_path, codec="libx264", audio_codec="aac")
        else:
            subclip.write_audiofile(segment_path)

        # Get Embedding for this specific segment
        with open(segment_path, "rb") as f:
            bytes_data = base64.b64encode(f.read()).decode('utf-8')
            vector = get_multimodal_embedding(bytes_data, type=modality)

        # Push to ChromaDB
        collection.add(
            ids=[f"{modality}_{start_t}"],
            embeddings=[vector],
            metadatas=[{
                "type": modality,
                "path": segment_path,
                "start_time": start_t,
                "end_time": end_t,
                "source": "tesla_q4_earnings"
            }]
        )
        print(f"Indexed {modality} segment: {start_t}s to {end_t}s")


def ingest_separate_media(video_path, audio_path, segment_duration=30):
    """
    Processes separate video and audio files, ensuring they are 
    indexed into the same ChromaDB collection.
    """
    print("\n--- [INFO] Processing Separate Video and Audio Files ---")
    print(f"Video Path: {video_path}")
    print(f"Audio Path: {audio_path}")

    v_clip = VideoFileClip(video_path)
    a_clip = AudioFileClip(audio_path)
    
    # Use the shorter of the two to avoid errors
    print(f"Video Duration: {v_clip.duration}s, Audio Duration: {a_clip.duration}s")
    total_duration = min(v_clip.duration, a_clip.duration, 300) # Limit to 5 mins
    print(f"Processing up to {total_duration}s of media in {segment_duration}s segments.")
    
    os.makedirs("./multimodal_rag/data/processed_segments", exist_ok=True)

    for start_t in range(0, int(total_duration), segment_duration):
        print(f"\n--- [INFO] Processing segment starting at {start_t}s ---")
        end_t = min(start_t + segment_duration, total_duration)
        
        # 1. PROCESS AUDIO SEGMENT
        audio_segment_path = f"./multimodal_rag/data/processed_segments/audio_{start_t}_{end_t}.mp3"
        a_sub = a_clip.subclipped(start_t, end_t)
        a_sub.write_audiofile(audio_segment_path, bitrate="192k")
        
        # 2. PROCESS VIDEO SEGMENT (No audio to keep it light)
        video_segment_path = f"./multimodal_rag/data/processed_segments/video_{start_t}_{end_t}.mp4"
        v_sub = v_clip.subclipped(start_t, end_t).without_audio()
        v_sub.write_videofile(video_segment_path, codec="libx264", fps=1) # Low FPS for RAG slides

        # 3. MULTIMODAL EMBEDDING & PUSH
        # We embed the Audio segment (Speech Context)
        with open(audio_segment_path, "rb") as f:
            a_bytes = base64.b64encode(f.read()).decode('utf-8')
            a_vector = get_multimodal_embedding(a_bytes, type="audio")
            
        # We embed a representative frame from the Video (Visual Context)
        # In 2026, Nova allows embedding the whole segment, but a keyframe is faster for retrieval
        frame_path = f"./multimodal_rag/data/processed_segments/frame_{start_t}.png"

        # if the very last segment of your video is shorter than 2 seconds, save_frame will throw an error because t=2 will be out of bounds for that specific subclip.
        # If the last segment is shorter than 2 seconds, we can adjust t to be the midpoint of the segment or just slightly before the end to ensure it's within bounds.
        
        t = min(2, v_sub.duration - 0.1)  # Ensure t is within bounds
        v_sub.save_frame(frame_path, t=t) # Save frame at 2 seconds into the clip
        with open(frame_path, "rb") as f:
            v_bytes = base64.b64encode(f.read()).decode('utf-8')
            v_vector = get_multimodal_embedding(v_bytes, type="image")

        # Push both to ChromaDB
        print(f"Pushing audio and video segments for interval {start_t}-{end_t}s to ChromaDB...")
        collection.add(
            ids=[f"audio_{start_t}", f"video_{start_t}"],
            embeddings=[a_vector, v_vector],
            metadatas=[
                {"type": "audio", "path": audio_segment_path, "start": start_t, "end": end_t},
                {"type": "video", "path": video_segment_path, "frame_ref": frame_path, "start": start_t}
            ]
        )
        print(f"Indexed pair for interval {start_t}-{end_t}s")
    print("\n--- [INFO] Finished processing separate media files ---")


if __name__ == "__main__":

    ## Ingest text and image data here:
    # print("\n=== Starting Multimodal Ingestion Process ===")
    # run_ingestion()
    # print("\n=== Multimodal Ingestion Process Finished ===")


    ## Ingest audio/video data here: Not used this.
    # Process Video (which includes audio)
    # ingest_media_segments("data/raw_videos/tesla_q4_2024.mp4", modality="video")  # Not used here.


    ## Process separate video and audio files (if you have them)
    ingest_separate_media(
        video_path="./multimodal_rag/data/raw_videos/tesla_q4_2024.mp4",
        audio_path="./multimodal_rag/data/raw_audio/tesla_q4_2024.mp3"
    )
