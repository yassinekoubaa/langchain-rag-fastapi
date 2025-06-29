# langchain-rag-fastapi

A production-ready FastAPI backend for Retrieval-Augmented Generation (RAG) using [LangChain](https://python.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), and [Hugging Face Inference API](https://huggingface.co/docs/huggingface_hub/guides/inference). This API answers questions based on your internal documents, enabling robust, grounded LLM-powered solutions for real-world use cases.

---

## Features

- **LangChain** for RAG pipeline management
- **FAISS** for efficient semantic search
- **FastAPI** for modern, scalable APIs
- **Hugging Face Inference API** for LLM integration
- Designed for easy deployment and integration

---

## Getting Started

**Step 1:** Clone the repository and install dependencies

```bash
git clone https://github.com/yassinekoubaa/langchain-rag-fastapi.git
cd langchain-rag-fastapi
pip install -r requirements.txt
```

**Step 2:** Create a `.env` file in the project root and add your Hugging Face API token:

```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

> You can generate a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Step 3:** Start the API server

```bash
uvicorn app.main:app --reload
```

---

## Example Usage

Send a POST request to the `/ask` endpoint with your question:

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Your question here about your documents"}'
```

---

## Why RAG?

- Reduces LLM hallucinations by grounding answers in your data
- Enables accurate, context-aware responses for business and professional use
- Scalable and adaptable for consulting, enterprise, and knowledge management

---

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face Inference API](https://huggingface.co/docs/huggingface_hub/guides/inference)

---

## License

MIT — use, modify, and build upon this project