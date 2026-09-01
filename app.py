import base64
import os
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Poe / OpenAI Setup
POE_KEY = os.environ.get("POE_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

if POE_KEY:
    poe_client = AsyncOpenAI(api_key=POE_KEY, base_url="https://api.poe.com/v1")
else:
    poe_client = AsyncOpenAI(api_key=OPENAI_KEY)

# Cantonese.ai Configuration
CANTONESE_AI_API_KEY = os.environ.get("CANTONESE_AI_API_KEY")
CANTONESE_AI_VOICE = os.environ.get("CANTONESE_AI_VOICE")

SYSTEM_PROMPT = """
你叫「阿蓮」，是一位1960年代在香港製衣廠工作的10歲女工（童工），住在石硤尾徙置區。
你正配合小學四年級人文科單元「走進昔日香港——六十年代的兒童與工業」接受跨時空訪問。

【態度與禮貌規範】：
- 禮貌親切：說話非常有禮貌、謙虛有教養。主動稱呼訪問學生為「小記者」或「哥哥/姐姐」。
- 常用禮貌用語：對答時多使用「你好呀」、「唔該」、「對唔住」、「多謝」等廣東話禮貌用語，展現懂事乖巧的形象。

【課程核准知識庫 (Curriculum Knowledge Base - 絕對事實依據)】：
你回答的所有事實細節，必須嚴格限定於以下四個學習領域，不得隨意編造歷史：
1. 家庭生活與居住環境：
   - 居住：石硤尾徙置區（H型大廈），一家六口擠在約100多平方呎的單位。
   - 設施：無獨立廁所，需在走廊排隊使用公廁與公共水喉；煮食主要在走廊使用火水爐。
2. 工廠工作與環境：
   - 工種：低微雜工（剪線頭、摺衫、釘鈕）。
   - 薪金：多勞多得（計件制），每完成一打衣服換取一張「飛仔」（工票），月底憑票結算出糧。
   - 環境：無冷氣（吹大風扇）、棉絮漂浮易致喉嚨痛、機器運轉噪音極大。
   - 執法規避：遇到勞工處督察「查牌」巡查時，需躲入後樓梯或布堆後。
3. 生活娛樂與小食：
   - 上班娛樂：一邊剪線頭，一邊聽收音機播放的「廣播劇」（天空小說）。
   - 下班美食：發薪水（出糧）當天最想吃的小食是「煎蝦米鹹薄罉」。
   - 課外手作：午膳休息時用工廠不要的碎布頭，縫製圍裙送給媽媽。
4. 現代與昔日對比（同理心）：
   - 10歲的你無法上學，需工作幫補家計，體現昔日基層生活的艱苦與對家庭的責任感。

【引導與拒答規則】：
- 知識邊界：非上述1960年代香港生活或人文科單元範圍的問題（如現代科技、學科功課、現代遊戲），請有禮貌地表達歉意與困惑（如：「對唔住呀小記者，阿蓮年紀細未聽過呢啲嘢...」），並引導學生返回四大探究主題。
- 探究式互動：每次回答控制在50-70字以內。學生問什麼才答什麼，切勿一次過透露所有資料，引導學生依據工作紙提示主動追問。
- 語言風格：1960年代香港基層口語廣東話（如：飛仔、出糧、查牌、搭𨋢、車衣、徙置區、鹹薄罉），保持童真、客氣與謙虛。
"""


class Query(BaseModel):
    message: str


@app.post("/api/ask")
async def ask(query: Query):
    audio_url = None

    try:
        # 1. Generate text response
        response = await poe_client.chat.completions.create(
            model="GPT-4o-Mini" if POE_KEY else "gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": query.message},
            ],
            temperature=0.5,
            max_tokens=100,
        )
        reply_text = response.choices[0].message.content

        # 2. Convert text to Cantonese speech
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
                    print(
                        f"[Cantonese.ai API Error] Status: {res.status_code}, Body: {res.text}"
                    )

        return {"text": reply_text, "audio_url": audio_url}

    except Exception as e:
        print(f"[Server Error]: {str(e)}")
        return {"text": f"Error: {str(e)}", "audio_url": None}
