import base64
import os
import re
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

class TTSRequest(BaseModel):
    text: str

def prepare_tts_text(text: str) -> str:
    """Cleans LaTeX/math formatting and safely truncates text to ensure fast TTS generation."""
    # 1. Remove LaTeX brackets, slashes, and math symbols
    cleaned = re.sub(r'\\[\(\)\[\]]', '', text)
    cleaned = re.sub(r'[\$\\]', '', cleaned)
    
    # 2. Convert digits to Cantonese characters
    num_map = {
        '90': '九十', '180': '一百八十', '360': '三百六十',
        '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
    }
    for k, v in num_map.items():
        cleaned = cleaned.replace(k, v)

    # 3. Truncate TTS speech to first sentence / max 45 characters so sound loads under 3s
    if len(cleaned) > 45:
        match = re.search(r'^.{15,45}[！!。？?]', cleaned)
        if match:
            cleaned = match.group(0)
        else:
            cleaned = cleaned[:45]

    return cleaned.strip()


@app.post("/api/chat")
async def chat(query: Query):
    poe_key = os.environ.get("POE_API_KEY", "").strip()

    if not poe_key:
        return {"text": "錯誤：未設定 POE_API_KEY 環境變數。"}

    cleaned_messages = [
        {"role": msg.get("role", "user"), "content": str(msg.get("content", "")).strip()}
        for msg in query.messages
        if str(msg.get("content", "")).strip()
    ]

    if not cleaned_messages:
        return {"text": "請輸入提問內容。"}

    system_instruction = {
        "role": "system",
        "content": "你係小學數學老師「余主任」。請用簡短、親切嘅繁體廣東話回答，每次回答控制喺35個字以內，避免複雜數學符號。"
    }
    formatted_messages = [system_instruction] + cleaned_messages

    try:
        poe_client = AsyncOpenAI(
            api_key=poe_key,
            base_url="https://api.poe.com/v1",
            timeout=6.0
        )

        response = await poe_client.chat.completions.create(
            model="mathchatbotyu",
            messages=formatted_messages,
            temperature=0.3,
            max_tokens=70
        )

        reply_text = ""
        if response.choices:
            msg = response.choices[0].message
            reply_text = (msg.content or getattr(msg, "reasoning_content", None) or "").strip()

        if not reply_text:
            reply_text = "余主任暫時未有回應，請確認 Poe Bot 名稱及點數餘額。"

        return {"text": reply_text}

    except Exception as e:
        logger.error(f"[Poe Error]: {str(e)}")
        return {"text": f"Poe 連線失敗：{str(e)}"}


@app.post("/api/tts")
async def generate_tts(req: TTSRequest):
    cantonese_key = os.environ.get("CANTONESE_AI_API_KEY", "").strip()
    cantonese_voice = os.environ.get("CANTONESE_AI_VOICE", "").strip()

    if not cantonese_key or not req.text:
        return {"audio_url": None}

    tts_text = prepare_tts_text(req.text)

    try:
        tts_url = "https://cantonese.ai/api/tts"
        headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        payload = {
            "api_key": cantonese_key,
            "text": tts_text,
            "output_extension": "mp3",
        }
        if cantonese_voice:
            payload["voice_id"] = cantonese_voice

        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.post(tts_url, json=payload, headers=headers)
            if res.status_code == 200:
                audio_b64 = base64.b64encode(res.content).decode("utf-8")
                return {"audio_url": f"data:audio/mp3;base64,{audio_b64}"}
            else:
                logger.error(f"[TTS Failed]: {res.status_code} - {res.text}")
    except Exception as tts_err:
        logger.error(f"[TTS Exception]: {str(tts_err)}")

    return {"audio_url": None}
