import base64
import os
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Poe API Setup
POE_KEY = os.environ.get("POE_API_KEY")
poe_client = AsyncOpenAI(api_key=POE_KEY, base_url="https://api.poe.com/v1")

# Cantonese.ai Configuration
CANTONESE_AI_API_KEY = os.environ.get("CANTONESE_AI_API_KEY")
CANTONESE_AI_VOICE = os.environ.get("CANTONESE_AI_VOICE")

class Query(BaseModel):
    # Updated to receive the full conversation array from the frontend
    messages: List[Dict[str, str]]

@app.post("/api/ask")
async def ask(query: Query):
    audio_url = None

    try:
        # Send the conversation history directly to your Poe Custom Bot
        response = await poe_client.chat.completions.create(
            model="DirectorYuMathHK", # 🔴 替換為你在 Poe 上的自訂 Bot Handle
            messages=query.messages,
            temperature=0.3,
            max_tokens=150,
        )
        reply_text = response.choices[0].message.content

        # Convert text to Cantonese speech
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
                else:
                    print(f"[Cantonese.ai API Error] Status: {res.status_code}, Body: {res.text}")

        return {"text": reply_text, "audio_url": audio_url}

    except Exception as e:
        print(f"[Server Error]: {str(e)}")
        return {"text": f"Error: {str(e)}", "audio_url": None}
