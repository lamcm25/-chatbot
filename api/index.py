import asyncio
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

# 你的 Google Apps Script Webhook 網址
WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbxI-n1nmYW43zAo-fShO7jCx1azXbL0EUo4W3HHibYU5epakHByMGjinEvG95jOX_da0w/exec"

class Query(BaseModel):
    messages: List[Dict[str, str]]

class TTSRequest(BaseModel):
    text: str

async def send_log_to_google_sheet(user_id: str, user_msg: str, bot_msg: str):
    """背景非同步發送對話紀錄至 Google Apps Script"""
    if not WEBHOOK_URL:
        return
    try:
        payload = {
            "user_id": user_id,
            "user_msg": user_msg,
            "bot_msg": bot_msg
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(WEBHOOK_URL, json=payload)
    except Exception as e:
        logger.error(f"[Google Sheet Log Error]: {str(e)}")

def prepare_tts_text(text: str) -> str:
    """清理 LaTeX/數學符號並保留完整內容供 TTS 朗讀（移除長度限制截斷）"""
    # 1. 移除 LaTeX 括號與斜線等符號
    cleaned = re.sub(r'\\[\(\)\[\]]', '', text)
    cleaned = re.sub(r'[\$\\]', '', cleaned)
    
    # 2. 數字轉廣東話漢字
    num_map = {
        '90': '九十', '180': '一百八十', '360': '三百六十',
        '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
    }
    for k, v in num_map.items():
        cleaned = cleaned.replace(k, v)

    # 3. 回傳完整清理後的文字，確保結尾句子能被完整朗讀
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
        "content": (
            "你係小學數學老師「余主任」。\n"
            "【說話規則】：\n"
            "1. 對話開始時，如果學生未提供名字，請先親切打招呼並問學生名字（例如：「你好呀！我係余主任。請問你叫咩名呀？」）。\n"
            "2. 如果已經知道學生名字，請稱呼佢名字。\n"
            "3. 請用簡短、親切嘅繁體廣東話回答，每次回答控制喺35個字以內，避免複雜數學符號。"
        )
    }
    formatted_messages = [system_instruction] + cleaned_messages

    try:
        poe_client = AsyncOpenAI(
            api_key=poe_key,
            base_url="https://api.poe.com/v1",
            timeout=8.0
        )

        response = await poe_client.chat.completions.create(
            model="masterYuBotnew2",
            messages=formatted_messages,
            temperature=0.3,
            max_tokens=85
        )

        reply_text = ""
        if response.choices:
            msg = response.choices[0].message
            reply_text = (msg.content or getattr(msg, "reasoning_content", None) or "").strip()

        if not reply_text:
            reply_text = "余主任暫時未有回應，請確認 Poe Bot 名稱及點數餘額。"

        # 成功取得回應後，於背景將對話紀錄傳送至 Google Sheet
        user_last_msg = cleaned_messages[-1]["content"] if cleaned_messages else ""
        asyncio.create_task(send_log_to_google_sheet("Web_User", user_last_msg, reply_text))

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
