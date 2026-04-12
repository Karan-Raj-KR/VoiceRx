from fastapi import FastAPI, Request
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from groq import Groq
from dotenv import load_dotenv
import os, json, httpx

load_dotenv()

app = FastAPI()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
list(embedding_model.embed(["warmup"]))

groq_client = None

COLLECTION_NAME = "health_faqs"

@app.post("/search")
async def search(request: Request):
    body = await request.json()
    print("BODY:", json.dumps(body, indent=2))
    
    # Extract question from Vapi webhook format
    try:
        tool_calls = body["message"]["toolCalls"]
        args = tool_calls[0]["function"]["arguments"]
        question = args["question"]
    except Exception:
        question = body.get("question", "")

    vector = list(embedding_model.embed([question]))[0].tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=3
    ).points

    context = "\n".join(r.payload["text"] for r in results) if results else ""

    try:
        global groq_client
        if groq_client is None:
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"), http_client=httpx.Client(timeout=8.0))
        print(f"[DEBUG] Question: {question}")
        print(f"[DEBUG] Context: {context}")
        chat = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful health assistant. Using the following health knowledge base context, answer the patient's question in 2-3 sentences. Be specific to their symptoms. Always recommend consulting a doctor for serious concerns. Context: {context}"
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=150,
            timeout=5
        )
        answer = chat.choices[0].message.content
        print(f"[DEBUG] Groq response: {answer}")
    except Exception:
        answer = "For your concern, please consult a doctor or visit the nearest clinic."

    # Return in Vapi expected format
    try:
        tool_call_id = body["message"]["toolCalls"][0]["id"]
        return {
            "results": [{
                "toolCallId": tool_call_id,
                "result": answer
            }]
        }
    except Exception:
        return {"result": answer}