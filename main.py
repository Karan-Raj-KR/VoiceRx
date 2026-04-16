from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from groq import Groq
from dotenv import load_dotenv
import os, json, logging, httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

required_env = [
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "GROQ_API_KEY",
    "VAPI_PUBLIC_KEY",
    "VAPI_ASSISTANT_ID",
]
missing = [v for v in required_env if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing required env vars: {missing}")

app = FastAPI()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
list(embedding_model.embed(["warmup"]))

groq_client = None

COLLECTION_NAME = "health_faqs"


def verify_collection():
    try:
        collections = client.get_collections()
        return any(c.name == COLLECTION_NAME for c in collections.collections)
    except Exception as e:
        logger.error(f"Failed to verify collection: {e}")
        return False


@app.on_event("startup")
async def startup():
    if not verify_collection():
        logger.warning(
            f"Collection '{COLLECTION_NAME}' not found. Run ingest.py first."
        )
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' verified.")


@app.get("/")
def serve_index():
    return FileResponse("index.html")


@app.get("/health")
async def detailed_health():
    status = {"qdrant": "connected" if verify_collection() else "missing_collection"}
    return status


@app.post("/search")
async def search(request: Request):
    body = await request.json()
    logger.info(f"Incoming request: {json.dumps(body, indent=2)}")

    if not verify_collection():
        return JSONResponse(
            status_code=503,
            content={"error": "Knowledge base not initialized. Run ingest.py."},
        )

    try:
        tool_calls = body.get("message", {}).get("toolCalls", [])
        if tool_calls:
            args = json.loads(tool_calls[0]["function"]["arguments"])
            question = args.get("question", "")
        else:
            question = body.get("question", "")
    except (json.JSONDecodeError, KeyError, TypeError):
        question = body.get("question", "")

    if not question:
        return JSONResponse(status_code=400, content={"error": "No question provided"})

    try:
        vector = list(embedding_model.embed([question]))[0].tolist()
        results = client.query_points(
            collection_name=COLLECTION_NAME, query=vector, limit=3, with_payload=True
        ).points

        context = (
            "\n".join(r.payload.get("text", "") for r in results if r.payload)
            if results
            else ""
        )

        if not context:
            answer = "I don't have specific information about that. Please consult a doctor for accurate guidance."
        else:
            global groq_client
            if groq_client is None:
                groq_client = Groq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    http_client=httpx.Client(timeout=30.0),
                )

            chat = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a helpful health assistant. Using the following health knowledge base context, answer the patient's question in 2-3 sentences. Be specific to their symptoms. Always recommend consulting a doctor for serious concerns. Context:\n{context}",
                    },
                    {"role": "user", "content": question},
                ],
                max_tokens=150,
                temperature=0.7,
            )
            answer = chat.choices[0].message.content

        logger.info(f"Generated answer: {answer}")

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        answer = (
            "For your concern, please consult a doctor or visit the nearest clinic."
        )

    try:
        tool_call_id = body["message"]["toolCalls"][0]["id"]
        return {"results": [{"toolCallId": tool_call_id, "result": answer}]}
    except (KeyError, IndexError):
        return {"result": answer}


@app.get("/config")
def get_config():
    return {
        "publicKey": os.getenv("VAPI_PUBLIC_KEY", ""),
        "assistantId": os.getenv("VAPI_ASSISTANT_ID", ""),
    }


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    logger.info(f"Webhook event: {json.dumps(body, indent=2)}")

    event_type = body.get("type", "")
    conversation_id = body.get("conversation_id", "")
    metadata = body.get("metadata", {})

    if event_type == "conversation-start":
        logger.info(f"Conversation {conversation_id} started")
    elif event_type == "conversation-end":
        logger.info(f"Conversation {conversation_id} ended")
    elif event_type == "speech-update":
        transcript = body.get("transcript", {})
        logger.info(f"Speech update: {transcript}")
    elif event_type == "tool-calls":
        tool_calls = body.get("tool_calls", [])
        logger.info(f"Tool calls received: {len(tool_calls)}")
    elif event_type == "tool-call-response":
        tool_call_id = body.get("tool_call_id", "")
        result = body.get("result", "")
        logger.info(f"Tool {tool_call_id} returned: {result[:100]}")

    return {"status": "received", "conversation_id": conversation_id}


@app.post("/vapi/webhook")
async def vapi_webhook(request: Request):
    """Dedicated endpoint for Vapi server-side webhooks"""
    body = await request.json()
    logger.info(f"Vapi webhook: {json.dumps(body)}")

    message = body.get("message", {})
    msg_type = message.get("type", "") if message else ""

    if msg_type == "tool-calls":
        tool_calls = message.get("toolCalls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            args = json.loads(func.get("arguments", "{}"))
            question = args.get("question", "")
            if question:
                logger.info(f"Processing tool call: {func.get('name')}")

    return {"status": "ok"}
