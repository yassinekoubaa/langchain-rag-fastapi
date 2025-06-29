"""
Intermediate Retrieval-Augmented Generation (RAG) FastAPI project.

✅ Comprehensive retrieval - Finds relevant information from multiple document sections
✅ Accurate responses - No hallucination, only document-based information
✅ Smart query classification - Distinguishes between current/past projects and specific queries
✅ Targeted document filtering - Filters documents based on question context
✅ Professional formatting - Clean, structured responses
✅ Error handling - Proper HTTP status codes and error messages
✅ Debug capabilities - Endpoints to troubleshoot retrieval issues
✅ Web UI - Simple chat interface for user interaction

The system demonstrates intermediate RAG techniques:
- Query classification for context-aware processing
- Document filtering based on content type and status
- Adaptive prompt engineering for different question types
- Production-ready FastAPI architecture with comprehensive error handling
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import asyncio
import time
from typing import Optional, List, Dict, Any
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

# Load environment variables
load_dotenv()

# Configuration and validation
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN environment variable is required")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
client = InferenceClient(model=MODEL_ID, token=HUGGINGFACE_TOKEN)

# FastAPI app configuration
app = FastAPI(
    title="Intermediate RAG Document QA API",
    description="Production-ready intermediate RAG system with smart query classification and document filtering",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "chat", "description": "Main question answering endpoints"},
        {"name": "debug", "description": "Development and debugging tools"},
        {"name": "system", "description": "System health and status"},
        {"name": "ui", "description": "Web interface"}
    ]
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Templates configuration
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Enhanced Pydantic models
class AskRequest(BaseModel):
    """Request model for asking questions."""
    prompt: str = Field(
        ..., 
        min_length=2, 
        max_length=500, 
        description="Question about the documents or greeting"
    )
    
    @validator('prompt')
    def validate_and_clean_prompt(cls, v: str) -> str:
        """Validate and clean the input prompt."""
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "What projects are we working on right now?"
            }
        }

class AskResponse(BaseModel):
    """Response model for question answers."""
    answer: str = Field(..., description="AI-generated answer based on document content")
    processing_time: Optional[float] = Field(None, description="Time taken to process the request in seconds")
    query_type: Optional[str] = Field(None, description="Classified query type for debugging")
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "• ERP Rollout for Orion Industrial Solutions – Implementing a unified ERP system...",
                "processing_time": 2.34,
                "query_type": "current_projects"
            }
        }

class ErrorResponse(BaseModel):
    """Standardized error response model."""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Detailed error description")
    suggestions: Optional[List[str]] = Field(None, description="Suggested actions to resolve the error")

class DebugRetrievalResponse(BaseModel):
    """Response model for debug retrieval endpoint.""" 
    query: str
    query_classification: dict
    retrieved_documents: List[dict]
    filtered_documents: List[dict]
    count: int
    total_chunks: int

# Document processing and indexing
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dummy_data.txt")

def initialize_document_system():
    """Initialize the document processing and retrieval system."""
    if not os.path.exists(DATA_PATH):
        raise RuntimeError(f"Document file not found at {DATA_PATH}")
    
    try:
        # Load documents
        loader = TextLoader(DATA_PATH, encoding='utf-8')
        docs = loader.load()

        # Split into chunks with optimized parameters
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\nProject Name:", "\n\nTitle:", "\n\nQuestion:", "\n\n", "\n", " "]
        )
        chunks = splitter.split_documents(docs)

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 12}  # Increased for better filtering
        )

        print(f"✅ Successfully loaded {len(chunks)} document chunks for retrieval.")
        return chunks, retriever

    except Exception as e:
        raise RuntimeError(f"Failed to initialize document processing: {str(e)}")

# Initialize the system
chunks, retriever = initialize_document_system()

# Enhanced Query Classification System
class QueryClassifier:
    """Production-ready query classification for RAG optimization."""
    
    def __init__(self):
        self.classification_patterns = {
            "current_projects": {
                "keywords": ["current", "working", "active", "ongoing", "now", "currently", "projects"],
                "context_filters": ["project brief", "project name", "client:"],
                "exclude_sections": ["lessons learned", "key takeaway"],
                "prompt_type": "current_projects"
            },
            "project_lessons": {
                "keywords": ["takeaway", "lesson", "learned", "key findings", "insights"],
                "context_filters": ["lessons learned", "key takeaway"],
                "exclude_sections": ["project brief"],
                "prompt_type": "lessons_learned"
            },
            "specific_project": {
                "keywords": [
                    "hr transformation", "cloud security", "erp rollout", "erp", 
                    "customer experience", "supply chain", "orion", "meridian", "crestview"
                ],
                "context_filters": [],
                "exclude_sections": [],
                "prompt_type": "specific_project"
            },
            "policies": {
                "keywords": ["policy", "procedure", "remote", "work from home", "guideline", "rule"],
                "context_filters": ["policy", "title:", "purpose:"],
                "exclude_sections": ["project", "lessons learned"],
                "prompt_type": "policy"
            },
            "escalation": {
                "keywords": ["escalate", "urgent", "emergency", "critical", "issue"],
                "context_filters": ["escalation", "after business hours"],
                "exclude_sections": [],
                "prompt_type": "escalation"
            }
        }
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify query with confidence scoring and filtering rules."""
        query_lower = query.lower()
        
        # Check for specific project mentions first
        project_matches = []
        for pattern_name, config in self.classification_patterns.items():
            if pattern_name == "specific_project":
                for keyword in config["keywords"]:
                    if keyword in query_lower:
                        project_matches.append(keyword)
        
        # Score each classification type
        classifications = {}
        for pattern_name, config in self.classification_patterns.items():
            score = 0
            matched_keywords = []
            
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                classifications[pattern_name] = {
                    "score": score,
                    "confidence": min(score / len(config["keywords"]), 1.0),
                    "matched_keywords": matched_keywords,
                    "config": config
                }
        
        # Determine primary classification
        if not classifications:
            return {
                "type": "general",
                "confidence": 1.0,
                "specific_projects": project_matches,
                "config": {"prompt_type": "general", "context_filters": [], "exclude_sections": []}
            }
        
        # Get highest scoring classification
        primary_type = max(classifications.keys(), key=lambda x: classifications[x]["score"])
        primary_classification = classifications[primary_type]
        
        return {
            "type": primary_type,
            "confidence": primary_classification["confidence"],
            "matched_keywords": primary_classification["matched_keywords"],
            "specific_projects": project_matches,
            "config": primary_classification["config"],
            "all_classifications": classifications
        }

def filter_documents_by_classification(docs: List, classification: Dict[str, Any]) -> List:
    """Enhanced document filtering with improved project detection."""
    if classification["type"] == "general":
        return docs[:6]  # Return top documents for general queries
    
    config = classification["config"]
    filtered_docs = []
    
    # Special handling for current projects to ensure comprehensive coverage
    if classification["type"] == "current_projects":
        project_brief_docs = []
        other_relevant_docs = []
        
        for doc in docs:
            content_lower = doc.page_content.lower()
            
            # Skip lessons learned sections for current projects
            if any(exclude_term in content_lower for exclude_term in ["lessons learned", "key takeaway"]):
                continue
            
            # Prioritize project brief sections
            if any(filter_term in content_lower for filter_term in ["project name:", "project brief", "client:"]):
                project_brief_docs.append(doc)
            # Include other potentially relevant project documents
            elif any(project_indicator in content_lower for project_indicator in [
                "rollout", "transformation", "framework", "assessment", "optimization"
            ]):
                other_relevant_docs.append(doc)
        
        # Combine project briefs first, then other relevant docs
        filtered_docs = project_brief_docs + other_relevant_docs
        
        # Ensure we have enough documents for comprehensive coverage
        if len(filtered_docs) < 6:
            remaining_docs = [doc for doc in docs if doc not in filtered_docs][:6-len(filtered_docs)]
            filtered_docs.extend(remaining_docs)
    
    else:
        # Original filtering logic for other query types
        for doc in docs:
            content_lower = doc.page_content.lower()
            
            # Check if document should be excluded
            should_exclude = False
            for exclude_term in config.get("exclude_sections", []):
                if exclude_term in content_lower:
                    should_exclude = True
                    break
            
            if should_exclude:
                continue
            
            # Check if document matches context filters
            if config.get("context_filters"):
                matches_filter = False
                for filter_term in config["context_filters"]:
                    if filter_term in content_lower:
                        matches_filter = True
                        break
                
                if matches_filter:
                    filtered_docs.append(doc)
            else:
                # No specific filters, include if not excluded
                filtered_docs.append(doc)
            
            # Limit results for performance
            if len(filtered_docs) >= 6:
                break
    
    # Enhanced fallback with better document selection
    if not filtered_docs:
        return docs[:4]
    
    return filtered_docs[:8]  # Allow more documents for better coverage

# Utility functions
def handle_greeting(prompt: str) -> Optional[str]:
    """Handle common greetings with appropriate responses."""
    prompt_lower = prompt.lower().strip()
    
    greetings = {
        'hi': "Hello! I'm your document assistant. Ask me about projects, policies, or procedures.",
        'hello': "Hi there! I can help you find information from company documents. What would you like to know?",
        'hey': "Hey! I'm here to answer questions about your documents. Try asking about current projects or policies.",
        'good morning': "Good morning! Ready to help you with document-related questions.",
        'good afternoon': "Good afternoon! How can I assist you with document queries today?",
        'thanks': "You're welcome! Feel free to ask more questions about the documents.",
        'thank you': "You're welcome! I'm here to help with any document-related questions."
    }
    
    return greetings.get(prompt_lower)

def is_valid_question(prompt: str) -> bool:
    """Check if the prompt contains valid question indicators."""
    if len(prompt) <= 10:  # Short prompts might be greetings
        return True
    
    question_keywords = [
        'what', 'how', 'when', 'where', 'who', 'why', 'which', 
        'describe', 'explain', 'list', 'tell', 'show', 'can you'
    ]
    return any(keyword in prompt.lower() for keyword in question_keywords)

async def generate_contextual_response(prompt: str, context: str, classification: Dict[str, Any]) -> str:
    """Generate contextually appropriate responses based on query classification."""
    
    query_type = classification["config"]["prompt_type"]
    
    if query_type == "current_projects":
        llm_prompt = f"""You are a business analyst. List ALL current active projects from the documents.

DOCUMENTS:
{context}

QUESTION: {prompt}

INSTRUCTIONS:
- List ALL active projects mentioned in Project Briefs or project descriptions
- Format: • Project Name for Client - Brief description
- Include projects like ERP Rollout, Customer Experience Transformation, ESG Framework
- Focus on active/current projects, not completed ones from lessons learned
- Be comprehensive and include all projects found

ANSWER:"""

    elif query_type == "lessons_learned":
        # Extract specific project if mentioned
        specific_projects = classification.get("specific_projects", [])
        project_focus = f" specifically about {', '.join(specific_projects)}" if specific_projects else ""
        
        llm_prompt = f"""You are a business analyst. Extract key takeaways{project_focus} from the Lessons Learned section.

DOCUMENTS:
{context}

QUESTION: {prompt}

INSTRUCTIONS:
- Focus on lessons learned and key takeaways
- Provide actionable insights and best practices
- Format as bullet points
- Be specific and practical

ANSWER:"""

    elif query_type == "specific_project":
        specific_projects = classification.get("specific_projects", [])
        project_name = specific_projects[0] if specific_projects else "the mentioned project"
        
        llm_prompt = f"""You are a business analyst. Provide information about {project_name}.

DOCUMENTS:
{context}

QUESTION: {prompt}

INSTRUCTIONS:
- Focus on the specific project mentioned in the question
- Provide relevant details from both project briefs and lessons learned if available
- Be comprehensive but concise
- Format clearly with bullet points if appropriate

ANSWER:"""

    elif query_type == "policy":
        llm_prompt = f"""You are a business analyst. Provide clear policy information.

DOCUMENTS:
{context}

QUESTION: {prompt}

INSTRUCTIONS:
- Extract specific policy details
- Format as bullet points for clarity
- Be precise and actionable
- Focus on the specific policy requirements

ANSWER:"""

    elif query_type == "escalation":
        llm_prompt = f"""You are a business analyst. Provide clear escalation procedures.

DOCUMENTS:
{context}

QUESTION: {prompt}

INSTRUCTIONS:
- Provide step-by-step escalation process
- Format as numbered or bulleted list
- Be specific about roles and timelines
- Focus on urgent issue handling

ANSWER:"""

    else:  # general
        llm_prompt = f"""You are a business analyst. Answer using the document information.

DOCUMENTS:
{context}

QUESTION: {prompt}

INSTRUCTIONS:
- Provide accurate information from documents
- Be concise and professional
- Use bullet points when appropriate

ANSWER:"""
    
    # Generate response with optimized parameters
    max_retries = 3
    timeout_seconds = 30
    
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.text_generation,
                    prompt=llm_prompt,
                    max_new_tokens=250,  # Increased for comprehensive project listing
                    temperature=0.05,    # Very low for factual accuracy
                    do_sample=True,
                    top_p=0.85,
                    repetition_penalty=1.15
                ),
                timeout=timeout_seconds
            )
            return response.strip()
            
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": "Request timeout",
                        "detail": f"Response took longer than {timeout_seconds} seconds",
                        "suggestions": ["Try a simpler question", "Try again in a moment"]
                    }
                )
            await asyncio.sleep(1)
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": "Model connection failed",
                        "detail": f"Failed after {max_retries} attempts: {str(e)}",
                        "suggestions": ["Check connection", "Try again later"]
                    }
                )
            await asyncio.sleep(2)

def clean_response(response: str) -> str:
    """Enhanced response cleaning."""
    # Remove verbose explanatory phrases
    unwanted_phrases = [
        "Note that", "However, if you interpret", "Therefore, it has been included",
        "The question asks for", "which implies that", "In such a case, the correct",
        "Please let me know", "Best regards", "Business Analyst", "END OF ANSWER",
        "The above response format", "Note: The question asked about",
        "Also, note that", "which could indicate either", "The final answer is:"
    ]
    
    for phrase in unwanted_phrases:
        if phrase in response:
            response = response.split(phrase)[0].strip()
    
    # Clean up incomplete sentences
    if response.endswith('.'):
        return response
    elif '.' in response:
        # Keep only complete sentences
        sentences = response.split('.')
        complete_sentences = [s.strip() for s in sentences[:-1] if s.strip()]
        return '. '.join(complete_sentences) + '.'
    
    return response.rstrip() + '.'

# Initialize query classifier
query_classifier = QueryClassifier()

# API Endpoints

@app.get("/", tags=["system"], summary="API information and navigation")
async def root():
    """Root endpoint providing API information and available interfaces."""
    return {
        "message": "Intermediate RAG Document Assistant API",
        "version": "1.1.0",
        "features": [
            "Smart query classification",
            "Context-aware document filtering", 
            "Adaptive prompt engineering",
            "Production-ready FastAPI architecture"
        ],
        "interfaces": {
            "web_ui": "/ui",
            "api_docs": "/docs",
            "redoc_docs": "/redoc",
            "health_check": "/health"
        },
        "status": "operational",
        "documents_loaded": len(chunks)
    }

@app.get("/ui", tags=["ui"], response_class=HTMLResponse, summary="Web chat interface")
async def get_ui(request: Request):
    """Serve the web-based chat interface for interacting with the RAG system."""
    try:
        return templates.TemplateResponse("chat.html", {"request": request})
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load web interface: {str(e)}"
        )

@app.post(
    "/ask",
    tags=["chat"],
    response_model=AskResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"}
    },
    summary="Ask a question about internal documents"
)
async def ask_question(request: AskRequest):
    """
    Enhanced endpoint with intelligent query classification and document filtering.
    
    Features:
    - Smart query classification for context-aware processing
    - Document filtering based on question type
    - Adaptive prompt engineering for different scenarios
    - Distinguishes between current/past projects automatically
    """
    start_time = time.time()
    
    try:
        # Handle greetings first
        greeting_response = handle_greeting(request.prompt)
        if greeting_response:
            return AskResponse(
                answer=greeting_response,
                processing_time=round(time.time() - start_time, 3),
                query_type="greeting"
            )
        
        # Validate question format
        if not is_valid_question(request.prompt):
            return AskResponse(
                answer="I need a more specific question. Try asking about projects, policies, or procedures using words like 'what', 'how', or 'when'.",
                processing_time=round(time.time() - start_time, 3),
                query_type="invalid"
            )

        # Classify query for targeted processing
        classification = query_classifier.classify_query(request.prompt)
        
        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(request.prompt)
        
        if not relevant_docs:
            return AskResponse(
                answer="I couldn't find relevant information to answer your question. Please try asking about projects, policies, or procedures.",
                processing_time=round(time.time() - start_time, 3),
                query_type=classification["type"]
            )
        
        # Filter documents based on classification
        filtered_docs = filter_documents_by_classification(relevant_docs, classification)
        
        # Further filter for specific projects if mentioned
        if classification.get("specific_projects"):
            filtered_docs = extract_specific_project_content(filtered_docs, classification["specific_projects"])
        
        # Combine filtered context
        context = "\n---\n".join([doc.page_content for doc in filtered_docs])
        
        # Generate contextual response
        response = await generate_contextual_response(request.prompt, context, classification)
        
        # Clean response
        answer = clean_response(response)
        
        # Fallback if response is empty
        if not answer:
            answer = "I found relevant documents but couldn't generate a clear answer. Please try rephrasing your question."
        
        processing_time = round(time.time() - start_time, 3)
        
        return AskResponse(
            answer=answer,
            processing_time=processing_time,
            query_type=classification["type"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return AskResponse(
            answer="I encountered an issue processing your request. Please try again or contact support if the problem persists.",
            processing_time=round(time.time() - start_time, 3),
            query_type="error"
        )

@app.post(
    "/debug/retrieve",
    tags=["debug"],
    response_model=DebugRetrievalResponse,
    summary="Debug enhanced document retrieval"
)
async def debug_retrieve(request: AskRequest):
    """
    Enhanced debug endpoint showing query classification and document filtering.
    """
    try:
        # Classify query
        classification = query_classifier.classify_query(request.prompt)
        
        # Get initial retrieval
        relevant_docs = retriever.get_relevant_documents(request.prompt)
        
        # Apply filtering
        filtered_docs = filter_documents_by_classification(relevant_docs, classification)
        
        if classification.get("specific_projects"):
            filtered_docs = extract_specific_project_content(filtered_docs, classification["specific_projects"])
        
        return DebugRetrievalResponse(
            query=request.prompt,
            query_classification=classification,
            retrieved_documents=[
                {
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": doc.metadata,
                    "length": len(doc.page_content)
                }
                for doc in relevant_docs
            ],
            filtered_documents=[
                {
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": doc.metadata,
                    "length": len(doc.page_content)
                }
                for doc in filtered_docs
            ],
            count=len(filtered_docs),
            total_chunks=len(chunks)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during enhanced retrieval debug: {str(e)}"
        )

@app.get("/debug/chunks", tags=["debug"], summary="Debug document chunks")
async def debug_chunks():
    """
    Debug endpoint to inspect how documents are chunked and indexed.
    """
    try:
        return {
            "total_chunks": len(chunks),
            "chunking_strategy": {
                "chunk_size": 800,
                "chunk_overlap": 100,
                "separators": ["\\n\\nProject Name:", "\\n\\nTitle:", "\\n\\nQuestion:", "\\n\\n", "\\n", " "]
            },
            "sample_chunks": [
                {
                    "index": i,
                    "content": chunk.page_content[:200] + ("..." if len(chunk.page_content) > 200 else ""),
                    "metadata": chunk.metadata,
                    "length": len(chunk.page_content)
                }
                for i, chunk in enumerate(chunks[:10])
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving chunk information: {str(e)}"
        )

@app.get("/health", tags=["system"], summary="System health check")
async def health_check():
    """
    Comprehensive health check endpoint.
    """
    try:
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "system_info": {
                "documents_loaded": len(chunks),
                "model": MODEL_ID,
                "retriever_type": "FAISS with MMR",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "rag_level": "intermediate"
            },
            "features": {
                "query_classification": "enabled",
                "document_filtering": "enabled",
                "contextual_prompting": "enabled",
                "smart_retrieval": "enabled"
            },
            "api_info": {
                "version": "1.1.0",
                "interfaces": ["REST API", "Web UI"],
                "endpoints": {
                    "primary": "/ask",
                    "debug": ["/debug/retrieve", "/debug/chunks"],
                    "interface": "/ui",
                    "health": "/health"
                }
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

# Exception handlers
@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    """Custom handler for validation errors following FastAPI best practices."""
    return HTTPException(
        status_code=422,
        detail={
            "error": "Validation error",
            "detail": "Please check your input format and try again",
            "suggestions": ["Ensure your question is between 2-500 characters"]
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)