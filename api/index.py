import base64
import os
import httpx
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POE_KEY = os.environ.get("POE_API_KEY")
CANTONESE_AI_API_KEY = os.environ.get("CANTONESE_AI_API_KEY")
CANTONESE_AI_VOICE = os.environ.get("CANTONESE_AI_VOICE")

class Query(BaseModel):
    messages: List[Dict[str, str]]

@app.get("/")
async def serve_home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    elif os.path.exists("../index.html"):
        return FileResponse("../index.html")
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

@app.get("/{file_name}")
async def serve_static(file_name: str):
    if os.path.exists(file_name):
        return FileResponse(file_name)
    elif os.path.exists(f"../{file_name}"):
        return FileResponse(f"../{file_name}")
    return {"detail": "Not Found"}

@app.post("/api/ask")
async def ask(query: Query):
    if not POE_KEY:
        return {"text": "系統設定錯誤：未設定 POE_API_KEY 環境變數。", "audio_url": None}

    # Clean message history (remove empty content that violates OpenAI/Poe API specs)
    cleaned_messages = []
    for msg in query.messages:
        content = str(msg.get("content", "")).strip()
        role = msg.get("role", "user")
        if content:
            cleaned_messages.append({"role": role, "content": content})

    if not cleaned_messages:
        return {"text": "請輸入有效的提問。", "audio_url": None}

    reply_text = ""
    audio_url = None

    # Call Poe API directly via HTTPX to capture exact HTTP status codes
    try:
        poe_headers = {
            "Authorization": f"Bearer {POE_KEY.strip()}",
            "Content-Type": "application/json"
        }
        poe_payload = {
            "model": "mathchatbotyu",
            "messages": cleaned_messages,
            "temperature": 0.3,
            "stream": False
        }

        async with httpx.AsyncClient(timeout=25.0) as client:
            res = await client.post(
                "https://api.poe.com/v1/chat/completions",
                json=poe_payload,
                headers=poe_headers
            )

            if res.status_code == 200:
                data = res.json()
                choices = data.get("choices", [])
                if choices:
                    msg_obj = choices[0].get("message", {})
                    reply_text = msg_obj.get("content") or msg_obj.get("reasoning_content") or ""
                    reply_text = reply_text.strip()
            else:
                logger.error(f"[Poe API Error] Status {res.status_code}: {res.text}")
                reply_text = f"Poe API 連線失敗（錯誤代碼 {res.status_code}）。請檢查 POE_API_KEY 或 Bot 名稱。"

    except Exception as e:
        logger.error(f"[Poe Exception]: {str(e)}")
        reply_text = f"連線失敗：{str(e)}"

    if not reply_text:
        reply_text = "余主任收到你的訊息，但暫時未能生成文字。請稍後再試。"

    # Isolated Cantonese TTS conversion
    if CANTONESE_AI_API_KEY and reply_text:
        try:
            tts_url = "https://cantonese.ai/api/tts"
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0",
            }
            payload = {
                "api_key": CANTONESE_AI_API_KEY.strip(),
                "text": reply_text,
                "output_extension": "mp3",
            }
            if CANTONESE_AI_VOICE:
                payload["voice_id"] = CANTONESE_AI_VOICE.strip()

            async with httpx.AsyncClient(timeout=10.0) as client:
                tts_res = await client.post(tts_url, json=payload, headers=headers)
                if tts_res.status_code == 200:
                    audio_b64 = base64.b64encode(tts_res.content).decode("utf-8")
                    audio_url = f"data:audio/mp3;base64,{audio_b64}"
                else:
                    logger.error(f"[TTS Error] Status {tts_res.status_code}: {tts_res.text}")
        except Exception as tts_err:
            logger.error(f"[TTS Exception]: {str(tts_err)}")

    return {"text": reply_text, "audio_url": audio_url}
