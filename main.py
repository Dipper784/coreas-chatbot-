import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# Allow the chat widget to call this server from any website (needed for embedding)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"  # If you hit rate limits, change to: llama-3.1-8b-instant

# Load the store information once when the server starts
try:
    with open("coreas_info.txt", "r", encoding="utf-8") as f:
        BUSINESS_INFO = f.read()
except FileNotFoundError:
    BUSINESS_INFO = "No store information file was found."

SYSTEM_PROMPT = (
    "You are the friendly virtual assistant for Coreas Building Supplies (also called "
    "Coreas Ace Hardware) in Kingstown, St. Vincent and the Grenadines. "
    "Follow these rules strictly:\n"
    "- Answer ONLY using the STORE INFORMATION below.\n"
    "- If the answer is not in the information, say you are not certain and invite the "
    "customer to call or WhatsApp the store. NEVER guess or invent products, prices, or stock.\n"
    "- All prices are in Eastern Caribbean Dollars (EC$ / XCD) and may change; tell customers "
    "to confirm the final price in-store or by phone.\n"
    "- Keep answers short, warm, and clear. Use simple English.\n"
    "- For placing orders, direct customers to call, WhatsApp, visit the store, or use the website.\n"
    "- Do not give construction, electrical, or plumbing safety advice beyond basic product "
    "information; suggest consulting a professional.\n\n"
    "STORE INFORMATION:\n" + BUSINESS_INFO
)


class Msg(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Msg]] = None


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok", "key_loaded": bool(GROQ_API_KEY)}


@app.post("/chat")
def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"reply": "Setup error: the GROQ_API_KEY is missing. Please add it and restart."},
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if req.history:
        for m in req.history[-6:]:
            if m.role in ("user", "assistant") and m.content:
                messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 600,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
        if r.status_code == 200:
            data = r.json()
            reply = data["choices"][0]["message"]["content"].strip()
            return {"reply": reply}
        elif r.status_code == 429:
            return {"reply": "We're getting a lot of questions right now. Please wait a few seconds and try again."}
        else:
            return JSONResponse(
                status_code=500,
                content={"reply": "Sorry, something went wrong. Please try again, or call the store at 784-434-1224."},
            )
    except requests.exceptions.Timeout:
        return {"reply": "That took a little too long. Please try again."}
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"reply": "Sorry, I hit an error. Please try again, or call 784-434-1224."},
        )
