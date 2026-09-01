import base64
import os
import logging
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
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

class Query(BaseModel):
    messages: List[Dict[str, str]]

@app.post("/api/ask")
async def ask(query: Query):
    poe_key = os.environ.get("POE_API_KEY", "").strip()
    cantonese_key = os.environ.get("CANTONESE_AI_API_KEY", "").strip()
    cantonese_voice = os.environ.get("CANTONESE_AI_VOICE", "").strip()

    if not poe_key:
        return {"text": "錯誤：未設定 POE_API_KEY 環境變數。", "audio_url": None}

    # Add system prompt to force short Cantonese responses for fast TTS
    system_instruction = {
        "role": "system",
        "content": "你係小學數學老師余主任。請用親切廣東話回答，每次回答請保持簡短（30字以內），方便學生理解。"
    }

    cleaned_messages = [
        {"role": msg.get("role", "user"), "content": str(msg.get("content", "")).strip()}
        for msg in query.messages
        if str(msg.get("content", "")).strip()
    ]

    if not cleaned_messages:
        return {"text": "請輸入提問內容。", "audio_url": None}

    full_messages = [system_instruction] + cleaned_messages

    reply_text = ""
    audio_url = None

    try:
        poe_client = AsyncOpenAI(
            api_key=poe_key,
            base_url="https://api.poe.com/v1",
            timeout=5.0
        )

        response = await poe_client.chat.completions.create(
            model="mathchatbotyu",
            messages=full_messages,
            temperature=0.3,
            max_tokens=150
        )

        if response.choices:
            msg = response.choices[0].message
            reply_text = (msg.content or getattr(msg, "reasoning_content", None) or "").strip()

    except Exception as e:
        logger.error(f"[Poe API Error]: {str(e)}")
        return {"text": f"Poe 連線失敗：{str(e)}", "audio_url": None}

    if not reply_text:
        reply_text = "余主任暫時未有回應，請確認 Poe Bot 名稱及點數餘額。"

    # TTS Generation: limit text length to ensure generation completes within 4 seconds
    if cantonese_key:
        try:
            # Truncate text for TTS if it exceeds 60 characters to prevent Vercel timeout
            tts_text = reply_text[:60] if len(reply_text) > 60 else reply_text

            tts_url = "https://cantonese.ai/api/tts"
            headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
            payload = {
                "api_key": cantonese_key,
                "text": tts_text,
                "output_extension": "mp3",
            }
            if cantonese_voice:
                payload["voice_id"] = cantonese_voice

            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.post(tts_url, json=payload, headers=headers)
                if res.status_code == 200:
                    audio_b64 = base64.b64encode(res.content).decode("utf-8")
                    audio_url = f"data:audio/mp3;base64,{audio_b64}"
                else:
                    logger.error(f"[TTS Failed]: Status {res.status_code}, Response: {res.text}")
        except Exception as tts_err:
            logger.error(f"[TTS Exception]: {str(tts_err)}")

    return {"text": reply_text, "audio_url": audio_url}
