import base64
import os
import httpx
import logging
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

POE_KEY = os.environ.get("POE_API_KEY")
CANTONESE_AI_API_KEY = os.environ.get("CANTONESE_AI_API_KEY")
CANTONESE_AI_VOICE = os.environ.get("CANTONESE_AI_VOICE")

poe_client = AsyncOpenAI(
    api_key=POE_KEY if POE_KEY else "dummy_key", 
    base_url="https://api.poe.com/v1"
)

class Query(BaseModel):
    messages: List[Dict[str, str]]

@app.post("/api/ask")
async def ask(query: Query):
    if not POE_KEY:
        return {"text": "系統設定錯誤：未設定 POE_API_KEY。", "audio_url": None}

    audio_url = None

    try:
        response = await poe_client.chat.completions.create(
            model="mathchatbotyu",
            messages=query.messages,
            temperature=0.3,
            max_tokens=250,
        )
        reply_text = response.choices[0].message.content

        if CANTONESE_AI_API_KEY:
            tts_url = "https://cantonese.ai/api/tts"
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0",
            }
            payload = {
                "api_key": CANTONESE_AI_API_KEY,
                "text": reply_text,
                "output_extension": "mp3",
            }
            if CANTONESE_AI_VOICE:
                payload["voice_id"] = CANTONESE_AI_VOICE

            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(tts_url, json=payload, headers=headers)
                if res.status_code == 200:
                    audio_b64 = base64.b64encode(res.content).decode("utf-8")
                    audio_url = f"data:audio/mp3;base64,{audio_b64}"

        return {"text": reply_text, "audio_url": audio_url}

    except Exception as e:
        logger.error(f"[Server Error]: {str(e)}")
        return {"text": f"余主任現時繁忙，請稍後再試。（錯誤訊息：{str(e)}）", "audio_url": None}
