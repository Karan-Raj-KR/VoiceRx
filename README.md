# VoiceRx

AI-powered voice health assistant for patients who can't access doctors.

## Tech Stack

- **Vapi** — Voice AI (call handling, speech-to-text, text-to-speech)
- **Qdrant** — Vector database for semantic FAQ retrieval
- **Groq LLM** — `llama-3.1-8b-instant` for synthesizing answers
- **FastAPI** — Webhook server for Vapi tool calls
- **FastEmbed** — Local embeddings (`BAAI/bge-small-en-v1.5`)

## How It Works

```
Patient speaks
     │
     ▼
Vapi (speech-to-text)
     │
     ▼
FastAPI /search webhook
     │
     ▼
Qdrant RAG (top 3 semantically similar FAQs)
     │
     ▼
Groq LLM (synthesizes answer using retrieved context)
     │
     ▼
Vapi (text-to-speech)
     │
     ▼
Patient hears response
```

## Architecture

```
Patient speaks → Vapi (STT) → Tool Call → FastAPI /search endpoint → FastEmbed (embed query) → Qdrant (semantic search, top 3 results) → Groq LLM (synthesize answer with context) → Vapi (TTS) → Patient hears response
```

## Setup

### 1. Install dependencies

```bash
pip install fastapi uvicorn qdrant-client fastembed groq httpx python-dotenv
```

### 2. Set environment variables

Create a `.env` file in the project root:

```
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

### 3. Ingest health FAQs into Qdrant

```bash
python ingest.py
```

### 4. Start the server

```bash
uvicorn main:app --reload --port 8000
```

The webhook will be available at `http://localhost:8000/search`.

---

Built for **HackBLR 2026**
