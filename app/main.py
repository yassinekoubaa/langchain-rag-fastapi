"""
Basic Retrieval-Augmented Generation (RAG) FastAPI project.

This API demonstrates a minimal RAG pipeline:
- Loads and splits internal documents into chunks.
- Uses embeddings and a vector store (FAISS) for semantic retrieval.
- Retrieves relevant document chunks for each user question.
- Passes the retrieved context and question to a Hugging Face LLM for answer generation.

It's "basic" because:
- It uses a single text file as the document source.
- Retrieval is based on simple semantic similarity (no advanced ranking or filtering).
- The LLM prompt is a straightforward concatenation of context and question.
- No advanced features like streaming, feedback, or multi-turn memory.

Endpoints:
- POST /ask: Ask a question about the internal documents.
- GET /: Health check.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is not set or is empty.")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
client = InferenceClient(model=MODEL_ID, token=token)

app = FastAPI(
    title="LLM Inference API",
    description="A FastAPI endpoint for text generation using Hugging Face Inference API.",
    version="1.0.0"
)

class AskRequest(BaseModel):
    prompt: str

class AskResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AskResponse, summary="Generate text from a prompt")
async def ask(request: AskRequest):
    """
    Generate a response from the LLM given a prompt.
    """
    try:
        response = client.text_generation(
            prompt=request.prompt,
            max_new_tokens=100
        )
        return AskResponse(answer=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

@app.get("/", summary="Health check")
async def root():
    return {"message": "API is running"}