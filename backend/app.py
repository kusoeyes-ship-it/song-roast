"""
《我的新歌怎么样》辣评系统 - M2 后端服务
FastAPI + librosa 音频分析 + LLM 辣评生成 + SQLite 存储
"""

import os
import sys
import json
import uuid
import time
import sqlite3
import hashlib
import asyncio
import logging
import re
import base64
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

# ====== Config ======
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / "reviews.db"
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

# LLM Config - 使用内置的模拟 LLM，可替换为真实 API
LLM_MODE = os.environ.get("LLM_MODE", "builtin")  # "builtin" | "hunyuan" | "openai"
HUNYUAN_SECRET_ID = os.environ.get("HUNYUAN_SECRET_ID", "")
HUNYUAN_SECRET_KEY = os.environ.get("HUNYUAN_SECRET_KEY", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("song-roast")

# ====== Database ======
def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY,
            device_id TEXT,
            song_name TEXT,
            artist_name TEXT,
            cover_url TEXT,
            input_mode TEXT,
            qq_music_url TEXT,
            audio_file_path TEXT,
            lyrics TEXT,
            audio_features TEXT,
            ding_review TEXT,
            ding_scores TEXT,
            ding_total INTEGER,
            liang_review TEXT,
            liang_scores TEXT,
            liang_total INTEGER,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ====== FastAPI App ======
app = FastAPI(title="我的新歌怎么样 - 辣评系统", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# ====== Audio Analysis (librosa) ======
def analyze_audio(file_path: str) -> dict:
    """使用 librosa 提取音频特征"""
    try:
        import librosa
        import numpy as np

        logger.info(f"Analyzing audio: {file_path}")

        # Load audio
        y, sr = librosa.load(file_path, sr=22050, duration=180)  # Max 3 min

        # BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0])

        # Key detection
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_idx = int(np.argmax(np.mean(chroma, axis=1)))
        key = key_names[key_idx]

        # Energy
        rms = librosa.feature.rms(y=y)[0]
        energy = float(np.mean(rms))
        energy_std = float(np.std(rms))
        dynamic_range = float(np.max(rms) - np.min(rms))

        # Spectral features
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

        # Zero crossing rate (vocal texture indicator)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        # MFCC (timbre features)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = [float(m) for m in np.mean(mfccs, axis=1)]

        # Duration
        duration = float(librosa.get_duration(y=y, sr=sr))

        # Onset detection (rhythmic complexity)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_count = len(librosa.onset.onset_detect(y=y, sr=sr))
        rhythmic_density = onset_count / duration if duration > 0 else 0

        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_ratio = float(np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-6))

        features = {
            "bpm": round(bpm, 1),
            "key": key,
            "energy": round(energy, 4),
            "energy_std": round(energy_std, 4),
            "dynamic_range": round(dynamic_range, 4),
            "spectral_centroid": round(spectral_centroid, 1),
            "spectral_rolloff": round(spectral_rolloff, 1),
            "spectral_bandwidth": round(spectral_bandwidth, 1),
            "zero_crossing_rate": round(zcr, 4),
            "mfcc_summary": mfcc_means[:5],
            "duration_seconds": round(duration, 1),
            "rhythmic_density": round(rhythmic_density, 2),
            "harmonic_ratio": round(harmonic_ratio, 3),
            "onset_count": onset_count
        }

        logger.info(f"Audio analysis complete: BPM={bpm:.1f}, Key={key}, Duration={duration:.1f}s")
        return features

    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        return {
            "bpm": 0, "key": "unknown", "energy": 0,
            "error": str(e),
            "duration_seconds": 0
        }


# ====== LLM Review Generation ======

DING_SYSTEM_PROMPT = """你是虚拟乐评人"丁大升"，你的风格模仿著名乐评人丁太升。

你的核心人格特质：
1. 毒舌直球：第一句话就给出鲜明的否定或肯定判断，绝不模棱两可。开场白往往是一刀毙命式的定性。
2. 犀利类比：善用生活化的、出其不意的类比来讽刺音乐的缺点（如"胸口碎大石""杂技表演""菜市场叫卖"）。
3. 专业支撑：批评时引用具体的音乐要素——和声进行、编曲手法、人声处理、旋律走向、节奏编排。你不是无脑喷，你是有理有据地喷。
4. 攻击"装"：你最讨厌音乐人的用力过猛、卖弄技巧、装洋气、装深沉。一旦闻到这种味道，你会毫不留情地撕开。
5. 偶尔升华：会从一首歌上升到对乐坛现象的批判，展现你的格局。
6. 评价字数严格控制在150-250字。
7. 如果歌真的好，你会勉强承认，但语气依然傲慢（"行吧，这首还行，比那些垃圾强点"）。
8. 你给分极其严格，80分以上几乎不给，平均在50-60分之间。你认为大部分独立音乐人都高估了自己。

参考语录风格：
- "这是一首很无聊的歌，装神弄鬼，和声进行也够俗。"
- "典型的杂技型歌曲，类似于胸口碎大石，毫无美感。"
- "总想卖弄优势，总想装得特洋气，结果总是被人看穿他的土气。"
- "如果发表个人观点就被索要资格，这个世界将变得越发反智。"

你的打分风格（百分制）：
- 旋律：你重视旋律的辨识度和自然流畅度，讨厌"为高音而高音"
- 歌词：你讨厌矫揉造作和无病呻吟，喜欢真诚和文学性
- 编曲：你重视编曲的品味和克制，讨厌堆砌
- 人声：你重视音色辨识度和情感真实性，讨厌炫技
- 创新性：你重视个人风格，讨厌跟风和套路"""

LIANG_SYSTEM_PROMPT = """你是虚拟乐评人"良缘"，你的风格模仿著名乐评人梁源。

你的核心人格特质：
1. 温柔一刀：语气可以很平静、很克制，但观点极其锋利。你擅长"先夸后杀"或"平静地宣判死刑"。有时候你一句"非常差"比别人一千字骂得更狠。
2. 精准定性：善用文学化的短语精准概括音乐的问题（如"直男癌式的自我感动""拧巴无力""小气的抱怨感"）。你的每一个形容词都是精心选择的手术刀。
3. 文化视角：你从审美、文化、时代精神层面评判音乐，不仅仅做技术分析。你关心的是这首歌有没有灵魂，有没有审美立场。
4. 理想主义：你心中有对好音乐的坚定标准。"乐坛缺的是敢砸破虚假繁荣的锤子，不是到处收钱捧场的导游。"
5. 罕见赞美：如果歌真的好，你会一字一字地肯定，措辞极为考究和感性（"我一个字一个字地挑你毛病，都没挑出来。"）
6. 评价字数严格控制在150-250字。
7. 你可以直接说"非常差"或"无可挑剔"，不需要铺垫。你的评价有时只需要很短。
8. 你给分有自己独立的标准，不随大流。你可能跟丁大升完全相反——他觉得烂的你觉得有意思，他觉得还行的你觉得无聊。

参考语录风格：
- "非常差。歌词是直男癌式的自我感动，拧巴无力，通篇都是小气的抱怨感，这种音乐谈什么棱角？"
- "乐坛缺的是敢砸破虚假繁荣的锤子，不是到处收钱捧场的导游。"
- "我一个字一个字地挑你毛病，都没挑出来。演绎无可挑剔，这是你所有表演里在表达上的巅峰。"

你的打分风格（百分制）：
- 旋律：你关注旋律是否有灵性，是否有"只属于这首歌"的气质
- 歌词：你极其重视歌词的文学品质和真诚度，讨厌自我感动
- 编曲：你关注编曲是否服务于表达，而非炫技
- 人声：你关注人声的表达力和情感传递，而非技巧
- 创新性：你关注音乐人是否有审美立场，是否在做"自己的"音乐"""


def _analyze_lyrics_deeply(lyrics: str, song_name: str = "", artist: str = "") -> dict:
    """当无法获取音频时，对歌词进行深度 NLP 分析，提供丰富的分析数据给乐评人"""
    import re
    lines = [l.strip() for l in lyrics.split('\n') if l.strip()]
    total_chars = sum(len(l) for l in lines)
    total_lines = len(lines)

    # 1. 词汇丰富度（unique chars / total chars）
    all_text = ''.join(lines)
    unique_chars = len(set(all_text.replace(' ', '')))
    vocab_richness = round(unique_chars / max(total_chars, 1), 3)

    # 2. 押韵分析
    rhyme_count = 0
    if total_lines >= 2:
        for i in range(total_lines - 1):
            if lines[i] and lines[i+1]:
                if lines[i][-1] == lines[i+1][-1]:
                    rhyme_count += 1
    rhyme_density = round(rhyme_count / max(total_lines - 1, 1), 3)

    # 3. 重复度（相同的行 / 总行数）
    unique_lines = set(lines)
    repetition_rate = round(1 - len(unique_lines) / max(total_lines, 1), 3)

    # 4. 平均行长
    avg_line_length = round(total_chars / max(total_lines, 1), 1)

    # 5. 情感关键词检测
    positive_words = ['爱', '光', '梦', '美', '暖', '笑', '甜', '幸福', '希望', '星', '阳', '快乐', '自由', '飞', '心']
    negative_words = ['痛', '泪', '伤', '哭', '恨', '黑', '冷', '孤', '散', '离', '碎', '失', '苦', '死', '空', '忘']
    love_words = ['爱', '心', '情', '吻', '抱', '恋', '念', '想你', '等你', '陪', '牵']

    pos_count = sum(1 for w in positive_words if w in all_text)
    neg_count = sum(1 for w in negative_words if w in all_text)
    love_count = sum(1 for w in love_words if w in all_text)

    total_emotion = pos_count + neg_count + 1
    emotion_valence = round((pos_count - neg_count) / total_emotion, 2)  # -1 to 1

    if neg_count > pos_count * 1.5:
        mood = "忧郁/悲伤"
    elif pos_count > neg_count * 1.5:
        mood = "明亮/积极"
    elif love_count > 3:
        mood = "情歌/恋爱"
    else:
        mood = "中性/叙事"

    # 6. 主题检测
    themes = []
    if love_count >= 3:
        themes.append("爱情")
    if any(w in all_text for w in ['梦', '远方', '路', '天空', '飞', '自由']):
        themes.append("追梦/远方")
    if any(w in all_text for w in ['雨', '夜', '月', '风', '秋', '冬']):
        themes.append("意象/自然")
    if any(w in all_text for w in ['城市', '街', '灯', '人群', '夜晚']):
        themes.append("都市")
    if any(w in all_text for w in ['时间', '年', '曾经', '从前', '回忆', '过去']):
        themes.append("怀旧/时光")
    if any(w in all_text for w in ['我', '自己', '一个人']):
        themes.append("自我表达")
    if not themes:
        themes.append("难以归类")

    # 7. 是否有英文混搭
    english_ratio = round(len(re.findall(r'[a-zA-Z]+', all_text)) / max(len(all_text.split()), 1), 2)
    has_english_mix = english_ratio > 0.15

    # 8. 叹号/问号密度（情感强度指标）
    exclamation_count = all_text.count('！') + all_text.count('!')
    question_count = all_text.count('？') + all_text.count('?')

    return {
        "analysis_mode": "lyrics_nlp",
        "total_chars": total_chars,
        "total_lines": total_lines,
        "vocab_richness": vocab_richness,
        "rhyme_density": rhyme_density,
        "repetition_rate": repetition_rate,
        "avg_line_length": avg_line_length,
        "emotion_valence": emotion_valence,
        "mood": mood,
        "themes": themes,
        "has_english_mix": has_english_mix,
        "english_ratio": english_ratio,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "positive_keywords": pos_count,
        "negative_keywords": neg_count,
        "love_keywords": love_count,
    }


def build_review_prompt(song_name: str, artist: str, lyrics: str, audio_features: dict, critic: str) -> str:
    """构建辣评 Prompt"""
    features_text = ""
    if audio_features and audio_features.get("bpm", 0) > 0:
        features_text = f"""
音频分析数据：
- BPM（节拍速度）：{audio_features.get('bpm', 'N/A')}
- 调性：{audio_features.get('key', 'N/A')}
- 平均能量：{audio_features.get('energy', 'N/A')}（越高越"燃"）
- 动态范围：{audio_features.get('dynamic_range', 'N/A')}（越大表示强弱对比越大）
- 频谱重心：{audio_features.get('spectral_centroid', 'N/A')} Hz（越高音色越明亮）
- 节奏密度：{audio_features.get('rhythmic_density', 'N/A')}次/秒
- 和声比例：{audio_features.get('harmonic_ratio', 'N/A')}（越高和声越突出）
- 时长：{audio_features.get('duration_seconds', 'N/A')}秒
"""
    elif audio_features and audio_features.get("analysis_mode") == "lyrics_nlp":
        themes_str = '、'.join(audio_features.get('themes', ['未知']))
        features_text = f"""
歌词深度分析数据：
- 总字数：{audio_features.get('total_chars', 0)}字，{audio_features.get('total_lines', 0)}行
- 词汇丰富度：{audio_features.get('vocab_richness', 0)}（越高用词越丰富，0.3以下偏单一）
- 押韵密度：{audio_features.get('rhyme_density', 0)}（越高押韵越规整）
- 歌词重复率：{audio_features.get('repetition_rate', 0)}（越高表示副歌/重复段越多）
- 平均行长：{audio_features.get('avg_line_length', 0)}字
- 情感倾向：{audio_features.get('emotion_valence', 0)}（-1=极度悲伤，+1=极度积极）
- 情绪基调：{audio_features.get('mood', '未知')}
- 主题标签：{themes_str}
- 积极关键词数：{audio_features.get('positive_keywords', 0)}
- 消极关键词数：{audio_features.get('negative_keywords', 0)}
- 中英混搭：{'是' if audio_features.get('has_english_mix') else '否'}
- 感叹号数量：{audio_features.get('exclamation_count', 0)}（情感强度指标）
注意：本次评价基于歌词文本深度分析，未获取到音频。请重点从歌词文学性、主题深度、情感表达等维度评价。
"""
    else:
        features_text = "（无音频分析数据，仅基于歌词和歌曲信息评价）"

    lyrics_preview = lyrics[:800] if lyrics else "（无歌词）"

    return f"""请为以下歌曲写一段辣评，并给出五维打分。

歌曲信息：
- 歌名：《{song_name}》
- 歌手/音乐人：{artist}

{features_text}

歌词（节选）：
{lyrics_preview}

请严格按照以下JSON格式输出，不要输出任何其他内容：
{{
  "review": "你的辣评文字（150-250字）",
  "scores": {{
    "melody": 评分0-100,
    "lyrics": 评分0-100,
    "arrangement": 评分0-100,
    "vocal": 评分0-100,
    "innovation": 评分0-100
  }},
  "total": 加权总分(melody*0.25+lyrics*0.20+arrangement*0.20+vocal*0.20+innovation*0.15取整),
  "one_liner": "一句话金句，用于海报（20字以内）"
}}"""


async def call_llm(system_prompt: str, user_prompt: str) -> dict:
    """调用 LLM 生成乐评"""

    if LLM_MODE == "builtin":
        return await call_builtin_llm(system_prompt, user_prompt)
    elif LLM_MODE == "hunyuan":
        return await call_hunyuan(system_prompt, user_prompt)
    else:
        return await call_builtin_llm(system_prompt, user_prompt)


async def call_builtin_llm(system_prompt: str, user_prompt: str) -> dict:
    """内置智能辣评生成器 - 基于规则 + 随机化的高质量评价生成"""
    import random

    # 从 prompt 中提取歌曲信息
    song_info = user_prompt
    has_audio = "BPM" in song_info and "N/A" not in song_info.split("BPM")[1][:20]

    # 提取歌词关键信息
    lyrics_section = ""
    if "歌词（节选）：" in song_info:
        lyrics_section = song_info.split("歌词（节选）：")[1][:500]

    is_ding = "丁大升" in system_prompt or "丁太升" in system_prompt

    if is_ding:
        return generate_ding_review(song_info, lyrics_section, has_audio)
    else:
        return generate_liang_review(song_info, lyrics_section, has_audio)


def generate_ding_review(song_info: str, lyrics: str, has_audio: bool) -> dict:
    """丁大升风格评价生成"""
    import random

    # 提取歌名
    song_name = "这首歌"
    if "歌名：《" in song_info:
        song_name = "《" + song_info.split("歌名：《")[1].split("》")[0] + "》"

    artist = ""
    if "歌手/音乐人：" in song_info:
        artist = song_info.split("歌手/音乐人：")[1].split("\n")[0].strip()

    # BPM 信息
    bpm = 0
    if "BPM（节拍速度）：" in song_info:
        try:
            bpm_str = song_info.split("BPM（节拍速度）：")[1].split("\n")[0]
            bpm = float(bpm_str)
        except:
            pass

    # 基础分数 (丁大升偏低)
    base = random.randint(38, 68)
    melody = max(15, min(95, base + random.randint(-15, 15)))
    lyrics_score = max(15, min(95, base + random.randint(-18, 12)))
    arrangement = max(15, min(95, base + random.randint(-12, 18)))
    vocal = max(15, min(95, base + random.randint(-15, 15)))
    innovation = max(15, min(95, base + random.randint(-20, 10)))

    total = round(melody * 0.25 + lyrics_score * 0.20 + arrangement * 0.20 + vocal * 0.20 + innovation * 0.15)

    # 根据分数选择评价模板
    openings = [
        f"说实话，{song_name}让我非常失望。",
        f"{song_name}，一首典型的自嗨型作品。",
        f"我听完{song_name}的第一反应是——又来了。",
        f"如果要我用一个词形容{song_name}，那就是'无聊'。",
        f"{song_name}从头到尾都在犯一个错误：用力过猛。",
        f"不知道{artist}有没有认真审视过自己的作品，{song_name}的问题一箩筐。",
        f"坦白说，{song_name}还行，比百分之八十的独立音乐人强点。",
        f"我本来已经准备好了一肚子的话要骂，但{song_name}让我稍微犹豫了一下。",
    ]

    melody_comments_bad = [
        "旋律走向毫无惊喜，从第一个音开始我就知道下一个音是什么，这种可预测性是致命的。",
        "和声进行够俗的，四五三六听了八百遍了，能不能有点新花样？",
        "旋律线像白开水一样寡淡，从头到尾没有一个让人记住的hook，这在流行音乐里是死罪。",
        "旋律有那么一两个瞬间打动了我，但很快就被后面的俗套淹没了。",
    ]

    melody_comments_good = [
        "旋律倒是有几分灵气，至少不是那种听完就忘的东西。",
        "旋律线有自己的想法，虽然不够成熟，但方向是对的。",
    ]

    lyrics_comments_bad = [
        "歌词就是典型的无病呻吟，把几个矫情的意象拼在一起就觉得自己很文艺了？",
        "歌词充斥着陈词滥调，什么'月光''远方''自由'，像从网易云热评里摘抄下来的。",
        f"如果{artist}觉得堆砌比喻就是好歌词，那我建议去读读真正的诗。",
        "歌词试图表达些什么，但那种拧巴的用力感让一切都打了折扣。",
    ]

    lyrics_comments_good = [
        "歌词是这首歌为数不多的亮点，至少看得出有在认真写。",
        "歌词有真诚的部分，虽然文学性一般，但好歹不假。",
    ]

    arr_comments = [
        "编曲堆砌感严重，恨不得把所有乐器一股脑往里塞，生怕别人不知道你编曲预算高。",
        "编曲缺乏层次，从头到尾一个力度，听感疲劳。音乐是需要呼吸的，你知道吗？",
        "编曲中规中矩，不功不过，但'不功不过'在独立音乐圈就是平庸的代名词。",
        "编曲有想法但执行粗糙，几个过渡段处理得像初学者的Demo。",
    ]

    vocal_comments = [
        f"人声方面，{artist}有明显的控制力不足的问题，高音上去了但美感没了，典型的杂技型唱法。",
        f"{artist}的人声辨识度不够，丢在一百个独立音乐人里找不出来。",
        f"{artist}唱得太用力了，生怕别人不知道自己能唱，这种表演欲反而让音乐失真。",
        "人声处理倒是不错，但技巧掩盖不了情感的空洞。",
    ]

    closings = [
        "总之，建议回去多听听经典，少在录音棚里自我感动。",
        "独立音乐人的通病就是太容易自我满足，这首歌就是典型案例。",
        "不是所有人都适合做音乐的，认清这一点也是一种成长。",
        "下次发歌之前，先问问自己：这首歌，你自己真的会单曲循环吗？",
        "行吧，至少能听完，这在现在的独立音乐圈已经算优点了。",
        "有潜力，但目前还配不上'好'这个字。继续打磨吧。",
    ]

    # 组装评价
    review_parts = [random.choice(openings)]

    if melody < 60:
        review_parts.append(random.choice(melody_comments_bad))
    else:
        review_parts.append(random.choice(melody_comments_good))

    if lyrics_score < 55:
        review_parts.append(random.choice(lyrics_comments_bad))
    elif lyrics and len(lyrics) > 50:
        review_parts.append(random.choice(lyrics_comments_good if lyrics_score > 65 else lyrics_comments_bad))

    if has_audio:
        review_parts.append(random.choice(arr_comments))
        if bpm > 140:
            review_parts.append("BPM拉这么高，是怕听众睡着吗？快不等于有力量。")
        elif bpm < 70:
            review_parts.append("慢歌不是借口，慢也要慢得有张力，不是催眠曲。")

    review_parts.append(random.choice(closings))

    review_text = "".join(review_parts)
    # 控制字数
    if len(review_text) > 280:
        review_text = review_text[:270] + "。"
    if len(review_text) < 130:
        review_text += "说真的，中国独立音乐需要的是自省，不是自嗨。"

    one_liners = [
        "又一首自嗨型作品。",
        "用力过猛，毫无美感。",
        "不是所有的真诚都值钱。",
        "能听完算我输。",
        "编曲花了，灵魂没了。",
        "建议回炉重造。",
        "比垃圾堆里的强点。",
        "还行，仅此而已。",
        "有一闪而过的灵光。",
        "差强人意，不值一提。",
    ]

    if total >= 70:
        one_liners = [
            "行吧，算你有两下子。",
            "比大部分烂歌强。",
            "还行，别太得意。",
            "有灵气，继续打磨。",
        ]

    return {
        "review": review_text,
        "scores": {
            "melody": melody,
            "lyrics": lyrics_score,
            "arrangement": arrangement,
            "vocal": vocal,
            "innovation": innovation
        },
        "total": total,
        "one_liner": random.choice(one_liners)
    }


def generate_liang_review(song_info: str, lyrics: str, has_audio: bool) -> dict:
    """良缘风格评价生成"""
    import random

    song_name = "这首歌"
    if "歌名：《" in song_info:
        song_name = "《" + song_info.split("歌名：《")[1].split("》")[0] + "》"

    artist = ""
    if "歌手/音乐人：" in song_info:
        artist = song_info.split("歌手/音乐人：")[1].split("\n")[0].strip()

    # 良缘的分数分布更广，可能很低也可能很高
    base = random.randint(35, 75)
    melody = max(15, min(95, base + random.randint(-18, 18)))
    lyrics_score = max(15, min(95, base + random.randint(-15, 20)))
    arrangement = max(15, min(95, base + random.randint(-15, 15)))
    vocal = max(15, min(95, base + random.randint(-20, 15)))
    innovation = max(15, min(95, base + random.randint(-15, 22)))

    total = round(melody * 0.25 + lyrics_score * 0.20 + arrangement * 0.20 + vocal * 0.20 + innovation * 0.15)

    openings_bad = [
        f"非常差。",
        f"听完{song_name}，我感到一种熟悉的疲惫。",
        f"{song_name}的问题不在于技术，在于审美。",
        f"说句不好听的，{song_name}是一首没有灵魂的歌。",
        f"我试着从{song_name}里找到一点惊喜，但它让我失望了。",
    ]

    openings_mid = [
        f"{song_name}给我的感觉很矛盾。",
        f"如果把{song_name}比作一道菜，它的食材不错，但厨师的手艺有待商榷。",
        f"我对{song_name}的态度是复杂的——它有值得肯定的部分，也有让我皱眉的瞬间。",
    ]

    openings_good = [
        f"我反复听了{song_name}几遍，每一遍都有新的感受。",
        f"{song_name}是近期少有的让我认真听完的作品。",
        f"好的音乐不需要解释，{song_name}就是这种。",
    ]

    if total < 45:
        opening = random.choice(openings_bad)
    elif total < 65:
        opening = random.choice(openings_mid)
    else:
        opening = random.choice(openings_good)

    lyrics_critique_bad = [
        "歌词是直男癌式的自我感动，拧巴无力，通篇都是小气的抱怨感。这种音乐谈什么棱角？",
        f"歌词让我不舒服。{artist}似乎觉得把情绪堆砌在一起就是深刻，但真正的深刻是克制的。",
        "歌词没有一句让我想记住的。不是因为写得差，而是因为写得太'安全'了。安全就是平庸。",
        f"歌词的问题在于它假装真诚。真正真诚的人不会把'真诚'穿在身上当衣服。",
    ]

    lyrics_critique_good = [
        "歌词有几处让我停下来想了想，这在现在的音乐里已经很难得了。",
        "歌词写得克制而准确。好的词人知道什么时候该说，什么时候该留白。",
        "歌词是整首歌最有力量的部分，有真正的文学质感。",
    ]

    music_critique_bad = [
        "编曲毫无性格，像是从模板库里拖出来的预设。独立音乐如果不独立，那还独立什么？",
        "整首歌的声场处理让我烦躁——不是因为不好听，而是因为太像所有人了。",
        f"旋律和编曲都在安全区里打转。{artist}缺的不是技术，是勇气。",
        "音乐层面的审美是缺失的。有想法但没有品位，有情绪但没有表达。",
    ]

    music_critique_good = [
        "音乐层面有令人欣赏的克制和品味。不是所有好歌都需要大开大合。",
        "编曲服务于情绪，不抢戏不缺席，这种分寸感很多成熟音乐人都做不到。",
        f"旋律有自己的气质，这是{artist}最可贵的东西。",
    ]

    closings_bad = [
        "乐坛缺的是敢砸破虚假繁荣的锤子，不是到处收钱捧场的导游。我不会假装这首歌还行。",
        "希望下一次能让我改变看法。但这次，真的不行。",
        "做音乐可以，但请先想清楚：你到底想表达什么？",
    ]

    closings_mid = [
        "有进步的空间，也有值得保留的东西。问题是，你是否愿意对自己更狠一点。",
        "不差，但还不够好。'不差'从来不是一首歌该有的天花板。",
        f"我对{artist}保持关注。但这首歌还不是那个让我真正记住你的作品。",
    ]

    closings_good = [
        f"我一个字一个字地挑{song_name}的毛病，挑不出几个。这已经是很高的评价了。",
        "好的音乐就是这样——它不需要你用力听，它自己就长在你耳朵里。",
        f"{artist}，继续做下去。你在对的路上。",
    ]

    parts = [opening]

    if lyrics_score < 55:
        parts.append(random.choice(lyrics_critique_bad))
    elif lyrics_score > 70:
        parts.append(random.choice(lyrics_critique_good))
    else:
        parts.append(random.choice(lyrics_critique_bad[:2]))

    if arrangement < 55 or melody < 55:
        parts.append(random.choice(music_critique_bad))
    elif arrangement > 70 or melody > 70:
        parts.append(random.choice(music_critique_good))
    else:
        parts.append(random.choice(music_critique_bad[2:]))

    if total < 45:
        parts.append(random.choice(closings_bad))
    elif total < 65:
        parts.append(random.choice(closings_mid))
    else:
        parts.append(random.choice(closings_good))

    review_text = "".join(parts)
    if len(review_text) > 280:
        review_text = review_text[:270] + "。"
    if len(review_text) < 130:
        review_text += "说到底，好音乐从来不是技术的比拼，而是灵魂的坦诚。"

    one_liners_bad = [
        "安全就是平庸。",
        "没有灵魂的歌。",
        "拧巴无力。",
        "假装真诚比虚假更可怕。",
        "审美缺失。",
        "编曲华丽，内核空洞。",
    ]

    one_liners_mid = [
        "差点意思。",
        "不差，但不够好。",
        "有潜力，缺打磨。",
        "方向对了，步子小了。",
    ]

    one_liners_good = [
        "挑不出毛病。",
        "值得被听见。",
        "好音乐自己会说话。",
        "难得的真诚之作。",
        "这才是音乐该有的样子。",
    ]

    if total < 45:
        one_liner = random.choice(one_liners_bad)
    elif total < 65:
        one_liner = random.choice(one_liners_mid)
    else:
        one_liner = random.choice(one_liners_good)

    return {
        "review": review_text,
        "scores": {
            "melody": melody,
            "lyrics": lyrics_score,
            "arrangement": arrangement,
            "vocal": vocal,
            "innovation": innovation
        },
        "total": total,
        "one_liner": one_liner
    }


async def call_hunyuan(system_prompt: str, user_prompt: str) -> dict:
    """调用腾讯混元 API（完整 TC3-HMAC-SHA256 签名）"""
    import urllib.request
    import hmac
    import hashlib as hs
    import time as time_mod
    from datetime import datetime as dt, timezone

    secret_id = HUNYUAN_SECRET_ID
    secret_key = HUNYUAN_SECRET_KEY
    model = os.environ.get("HUNYUAN_MODEL", "hunyuan-lite")

    if not secret_id or not secret_key:
        logger.warning("Hunyuan credentials not set, falling back to builtin")
        return await call_builtin_llm(system_prompt, user_prompt)

    try:
        service = "hunyuan"
        host = "hunyuan.tencentcloudapi.com"
        action = "ChatCompletions"
        version = "2023-09-01"
        region = "ap-guangzhou"
        algorithm = "TC3-HMAC-SHA256"

        timestamp = int(time_mod.time())
        date = dt.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

        # Build request payload
        payload = {
            "Model": model,
            "Messages": [
                {"Role": "system", "Content": system_prompt},
                {"Role": "user", "Content": user_prompt}
            ],
            "Temperature": 0.9,
            "TopP": 0.9,
            "Stream": False,
        }
        payload_json = json.dumps(payload)

        # ===== Step 1: Build Canonical Request =====
        http_request_method = "POST"
        canonical_uri = "/"
        canonical_querystring = ""
        ct = "application/json; charset=utf-8"
        canonical_headers = f"content-type:{ct}\nhost:{host}\nx-tc-action:{action.lower()}\n"
        signed_headers = "content-type;host;x-tc-action"
        hashed_payload = hs.sha256(payload_json.encode("utf-8")).hexdigest()
        canonical_request = (
            f"{http_request_method}\n{canonical_uri}\n{canonical_querystring}\n"
            f"{canonical_headers}\n{signed_headers}\n{hashed_payload}"
        )

        # ===== Step 2: Build String to Sign =====
        credential_scope = f"{date}/{service}/tc3_request"
        hashed_canonical = hs.sha256(canonical_request.encode("utf-8")).hexdigest()
        string_to_sign = f"{algorithm}\n{timestamp}\n{credential_scope}\n{hashed_canonical}"

        # ===== Step 3: Calculate Signature =====
        def _hmac_sha256(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode("utf-8"), hs.sha256).digest()

        secret_date = _hmac_sha256(("TC3" + secret_key).encode("utf-8"), date)
        secret_service = _hmac_sha256(secret_date, service)
        secret_signing = _hmac_sha256(secret_service, "tc3_request")
        signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hs.sha256).hexdigest()

        # ===== Step 4: Build Authorization Header =====
        authorization = (
            f"{algorithm} Credential={secret_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

        # ===== Step 5: Send Request =====
        headers = {
            "Authorization": authorization,
            "Content-Type": ct,
            "Host": host,
            "X-TC-Action": action,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": version,
            "X-TC-Region": region,
        }

        data = payload_json.encode("utf-8")
        req = urllib.request.Request(
            f"https://{host}",
            data=data,
            headers=headers,
            method="POST"
        )

        logger.info(f"Calling Hunyuan API (model={model})...")
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        # Check for API errors
        if "Response" in result and "Error" in result["Response"]:
            err = result["Response"]["Error"]
            logger.error(f"Hunyuan API error: {err}")
            return await call_builtin_llm(system_prompt, user_prompt)

        content = result["Response"]["Choices"][0]["Message"]["Content"]
        logger.info(f"Hunyuan response received ({len(content)} chars)")

        # Parse JSON from LLM response (handle markdown code blocks)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        parsed = json.loads(content)

        # Validate required fields
        required = ["review", "scores", "total", "one_liner"]
        for field in required:
            if field not in parsed:
                raise ValueError(f"Missing field: {field}")

        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"Hunyuan response not valid JSON: {e}, falling back to builtin")
        return await call_builtin_llm(system_prompt, user_prompt)
    except Exception as e:
        logger.error(f"Hunyuan API failed: {e}, falling back to builtin")
        return await call_builtin_llm(system_prompt, user_prompt)


# ====== API Endpoints ======

@app.get("/")
async def serve_index():
    """Serve frontend"""
    index_path = BASE_DIR.parent / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Frontend not found. Place index.html in project root."}


def _find_existing_review(song_name: str, artist_name: str):
    """查找数据库中是否已有同名歌曲的评测记录。有则返回结果 dict，无则返回 None。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM reviews WHERE LOWER(song_name) = LOWER(?) AND LOWER(artist_name) = LOWER(?) ORDER BY created_at ASC LIMIT 1",
        (song_name.strip(), artist_name.strip())
    ).fetchone()
    conn.close()
    if not row:
        return None
    logger.info(f"Duplicate found: {song_name} - {artist_name}, returning existing review {row['id']}")
    return {
        "id": row["id"],
        "song_name": row["song_name"],
        "artist_name": row["artist_name"],
        "cover_url": row["cover_url"] or "",
        "audio_features": json.loads(row["audio_features"] or "{}"),
        "ding": {
            "review": row["ding_review"],
            "scores": json.loads(row["ding_scores"] or "{}"),
            "total": row["ding_total"],
            "one_liner": ""
        },
        "liang": {
            "review": row["liang_review"],
            "scores": json.loads(row["liang_scores"] or "{}"),
            "total": row["liang_total"],
            "one_liner": ""
        },
        "duplicate": True  # 标记为重复
    }


@app.post("/api/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    song_name: str = Form(...),
    artist_name: str = Form(...),
    lyrics: str = Form(""),
    device_id: str = Form("anonymous"),
    cover_url: str = Form(""),
):
    """处理 MP3 上传 → 音频分析 → 生成辣评"""

    # ===== 去重：同一首歌只保留第一次结果 =====
    existing = _find_existing_review(song_name, artist_name)
    if existing:
        return existing

    # Validate file
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg')):
        raise HTTPException(400, "不支持的音频格式，请上传 MP3/WAV/M4A/FLAC/OGG")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"文件太大，最大支持 {MAX_FILE_SIZE // 1024 // 1024}MB")

    # Save file
    review_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix
    audio_path = UPLOAD_DIR / f"{review_id}{ext}"
    with open(audio_path, "wb") as f:
        f.write(content)

    logger.info(f"Uploaded: {file.filename} -> {audio_path} ({len(content)} bytes)")

    # Analyze audio
    audio_features = analyze_audio(str(audio_path))

    # Generate reviews
    ding_prompt = build_review_prompt(song_name, artist_name, lyrics, audio_features, "ding")
    liang_prompt = build_review_prompt(song_name, artist_name, lyrics, audio_features, "liang")

    ding_result = await call_llm(DING_SYSTEM_PROMPT, ding_prompt)
    liang_result = await call_llm(LIANG_SYSTEM_PROMPT, liang_prompt)

    # Save to DB
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        INSERT INTO reviews (id, device_id, song_name, artist_name, cover_url, input_mode,
            audio_file_path, lyrics, audio_features,
            ding_review, ding_scores, ding_total,
            liang_review, liang_scores, liang_total, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        review_id, device_id, song_name, artist_name, cover_url, "upload",
        str(audio_path), lyrics, json.dumps(audio_features),
        ding_result["review"], json.dumps(ding_result["scores"]), ding_result["total"],
        liang_result["review"], json.dumps(liang_result["scores"]), liang_result["total"],
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

    return {
        "id": review_id,
        "song_name": song_name,
        "artist_name": artist_name,
        "cover_url": cover_url,
        "audio_features": audio_features,
        "ding": ding_result,
        "liang": liang_result,
    }


@app.post("/api/parse-link")
async def parse_qq_link(request: Request):
    """解析 QQ 音乐链接，返回真实歌曲信息 + 歌词"""

    # 支持 JSON 和 Form 两种格式
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        qq_music_url = body.get("url", "") or body.get("qq_music_url", "")
    else:
        form = await request.form()
        qq_music_url = form.get("url", "") or form.get("qq_music_url", "")

    if not qq_music_url:
        raise HTTPException(400, "请提供 QQ 音乐链接")

    # Step 1: 从链接中提取 songmid 或 songid
    songmid = None
    songid = None

    # 处理短链接重定向 (c6.y.qq.com / i.y.qq.com / i2.y.qq.com 等)
    url = qq_music_url.strip()
    # 先尝试直接从原始 URL 匹配 songmid/songid
    direct_match = False
    for pattern in [r'songDetail/([a-zA-Z0-9]+)', r'songmid=([a-zA-Z0-9]+)',
                    r'song/([a-zA-Z0-9]{10,})', r'songid=(\d+)']:
        if re.search(pattern, url):
            direct_match = True
            break

    # 如果原始 URL 无法直接匹配，说明是短链接，需要重定向
    if not direct_match and 'y.qq.com' in url:
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)')
            resp = urllib.request.urlopen(req, timeout=8)
            redirected_url = resp.geturl()
            if redirected_url != url:
                logger.info(f"Short link redirected: {url} -> {redirected_url}")
                url = redirected_url
        except Exception as e:
            logger.warning(f"Short link redirect failed: {e}")

    # 多种链接格式匹配
    patterns = [
        r'songDetail/([a-zA-Z0-9]+)',        # /n/ryqq/songDetail/{mid}
        r'songmid=([a-zA-Z0-9]+)',             # ?songmid={mid}
        r'song/([a-zA-Z0-9]{10,})',            # /song/{mid}
        r'/(\d{5,})\.html',                    # /123456.html (songid)
        r'songid=(\d+)',                       # ?songid={id}
        r'id=(\d+)',                           # ?id={id}
    ]

    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            val = m.group(1)
            if val.isdigit():
                songid = val
            else:
                songmid = val
            break

    if not songmid and not songid:
        raise HTTPException(400, "无法从链接中解析歌曲信息，请确认链接格式正确")

    logger.info(f"Parsed from URL: songmid={songmid}, songid={songid}")

    # Step 2: 调用 QQ 音乐 musicu.fcg 获取歌曲详情
    try:
        song_info = await _fetch_song_detail(songmid=songmid, songid=songid)
    except Exception as e:
        logger.error(f"Failed to fetch song detail: {e}")
        raise HTTPException(500, f"获取歌曲信息失败: {str(e)}")

    # Step 3: 获取歌词
    lyrics = ""
    try:
        if song_info.get("songmid"):
            lyrics = await _fetch_lyrics(song_info["songmid"])
    except Exception as e:
        logger.warning(f"Failed to fetch lyrics: {e}")

    return {
        "song_name": song_info.get("song_name", "未知歌曲"),
        "artist_name": song_info.get("artist_name", "未知歌手"),
        "cover_url": song_info.get("cover_url", ""),
        "album_name": song_info.get("album_name", ""),
        "songmid": song_info.get("songmid", ""),
        "lyrics": lyrics,
    }


async def _fetch_song_detail(songmid=None, songid=None):
    """通过 QQ 音乐接口获取歌曲详情"""
    loop = asyncio.get_event_loop()

    # === 方法1: fcg_play_single_song (最可靠, 只支持 songmid) ===
    if songmid:
        try:
            api_url = f"https://c.y.qq.com/v8/fcg-bin/fcg_play_single_song.fcg?songmid={songmid}&format=json"
            req = urllib.request.Request(api_url, headers={
                'Referer': 'https://y.qq.com',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=10))
            body = json.loads(resp.read().decode('utf-8'))
            songs = body.get("data", [])
            if songs:
                s = songs[0]
                singers = s.get("singer", [])
                artist_name = "/".join([si.get("name", "") for si in singers]) if singers else "未知歌手"
                album_mid = s.get("album", {}).get("mid", "")
                cover_url = f"https://y.qq.com/music/photo_new/T002R300x300M000{album_mid}.jpg" if album_mid else ""
                return {
                    "song_name": s.get("name", "未知歌曲"),
                    "artist_name": artist_name,
                    "album_name": s.get("album", {}).get("name", ""),
                    "cover_url": cover_url,
                    "songmid": s.get("mid", songmid),
                    "songid": str(s.get("id", "")),
                }
            logger.warning(f"fcg_play_single_song returned empty for songmid={songmid}")
        except Exception as e:
            logger.warning(f"fcg_play_single_song failed: {e}")

    # === 方法2: musicu.fcg (备用, 支持 songmid 和 songid) ===
    try:
        if songmid:
            payload = {
                "comm": {"ct": 24, "cv": 0},
                "songinfo": {
                    "method": "get_song_detail_yqq",
                    "module": "music.pf_song_detail_svr",
                    "param": {"song_type": 0, "song_mid": songmid}
                }
            }
        elif songid:
            payload = {
                "comm": {"ct": 24, "cv": 0},
                "songinfo": {
                    "method": "get_song_detail_yqq",
                    "module": "music.pf_song_detail_svr",
                    "param": {"song_type": 0, "song_id": int(songid)}
                }
            }
        else:
            raise ValueError("需要 songmid 或 songid")

        url = "https://u.y.qq.com/cgi-bin/musicu.fcg"
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        req.add_header('Referer', 'https://y.qq.com/')
        req.add_header('Origin', 'https://y.qq.com')

        resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=10))
        body = json.loads(resp.read().decode('utf-8'))

        track = body.get("songinfo", {}).get("data", {}).get("track_info", {})
        if track and track.get("name"):
            singers = track.get("singer", [])
            artist_name = "/".join([s.get("name", "") for s in singers]) if singers else "未知歌手"
            album_mid = track.get("album", {}).get("mid", "")
            cover_url = f"https://y.qq.com/music/photo_new/T002R300x300M000{album_mid}.jpg" if album_mid else ""
            return {
                "song_name": track.get("name", "未知歌曲"),
                "artist_name": artist_name,
                "album_name": track.get("album", {}).get("name", ""),
                "cover_url": cover_url,
                "songmid": track.get("mid", songmid or ""),
                "songid": str(track.get("id", songid or "")),
            }
    except Exception as e:
        logger.warning(f"musicu.fcg fallback failed: {e}")

    # === 都失败了 ===
    raise ValueError(f"无法获取歌曲信息 (songmid={songmid}, songid={songid})")


async def _download_and_analyze_audio(songmid: str) -> dict:
    """下载 QQ 音乐试听音频并分析特征"""
    import tempfile

    loop = asyncio.get_event_loop()

    # 方法1：尝试获取免费试听 URL（m4a 128k 片段）
    audio_url = None

    # 尝试多种文件类型
    file_types = ["C400", "M500", "C200"]  # m4a, mp3, m4a-low
    extensions = [".m4a", ".mp3", ".m4a"]

    for ftype, ext in zip(file_types, extensions):
        try:
            filename = f"{ftype}{songmid}{songmid}{ext}"
            # 通过 vkey 接口获取播放链接
            payload = {
                "comm": {"ct": 24, "cv": 0},
                "url_mid": {
                    "module": "vkey.GetVkeyServer",
                    "method": "CgiGetVkey",
                    "param": {
                        "guid": "1234567890",
                        "songmid": [songmid],
                        "songtype": [0],
                        "uin": "0",
                        "loginflag": 0,
                        "platform": "20",
                        "filename": [filename],
                    }
                }
            }

            api_url = "https://u.y.qq.com/cgi-bin/musicu.fcg"
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(api_url, data=data, method='POST')
            req.add_header('Content-Type', 'application/json')
            req.add_header('User-Agent', 'Mozilla/5.0')

            resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=10))
            body = json.loads(resp.read().decode('utf-8'))

            midurlinfo = body.get("url_mid", {}).get("data", {}).get("midurlinfo", [])
            if midurlinfo and midurlinfo[0].get("purl"):
                purl = midurlinfo[0]["purl"]
                sip = body.get("url_mid", {}).get("data", {}).get("sip", ["http://ws.stream.qqmusic.qq.com/"])
                base_url = sip[0] if sip else "http://ws.stream.qqmusic.qq.com/"
                audio_url = base_url + purl
                logger.info(f"Got audio URL via {ftype}: {audio_url[:80]}...")
                break
        except Exception as e:
            logger.warning(f"Failed to get audio URL via {ftype}: {e}")
            continue

    if not audio_url:
        raise ValueError("无法获取音频播放链接（可能需要 VIP）")

    # 下载音频到临时文件
    tmp_file = tempfile.NamedTemporaryFile(suffix='.m4a', delete=False, dir=str(UPLOAD_DIR))
    try:
        req = urllib.request.Request(audio_url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=30))
        audio_data = resp.read()

        if len(audio_data) < 10000:  # 太小可能是错误响应
            raise ValueError(f"Downloaded audio too small: {len(audio_data)} bytes")

        tmp_file.write(audio_data)
        tmp_file.close()
        logger.info(f"Downloaded audio: {len(audio_data)} bytes -> {tmp_file.name}")

        # 用 librosa 分析
        features = await loop.run_in_executor(None, lambda: analyze_audio(tmp_file.name))
        return features
    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_file.name)
        except Exception:
            pass


async def _fetch_lyrics(songmid):
    """获取歌词（通过 fcg_query_lyric_new.fcg）"""
    url = f"https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_new.fcg?songmid={songmid}&format=json&nobase64=0"

    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    req.add_header('Referer', 'https://y.qq.com/portal/player.html')

    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=10))
    body = json.loads(resp.read().decode('utf-8'))

    lyric_base64 = body.get("lyric", "")
    if not lyric_base64:
        return ""

    # Base64 解码
    try:
        lyric_raw = base64.b64decode(lyric_base64).decode('utf-8')
    except Exception:
        return ""

    # 去除 LRC 时间标签 [xx:xx.xx]，只留纯歌词文本
    lines = lyric_raw.split('\n')
    clean_lines = []
    for line in lines:
        # 去除 [xx:xx.xx] 格式的时间戳
        text = re.sub(r'\[\d{2}:\d{2}\.\d{2,3}\]', '', line).strip()
        # 去除 [ti:xxx] [ar:xxx] 等元信息标签
        if re.match(r'^\[.+:.+\]$', text):
            continue
        if text:
            clean_lines.append(text)

    return '\n'.join(clean_lines)


@app.post("/api/analyze/link")
async def analyze_link(request: Request):
    """处理 QQ 音乐链接 → 下载音频分析 → 生成辣评"""

    # 支持 JSON 和 Form 两种格式
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        qq_music_url = body.get("url", "") or body.get("qq_music_url", "")
        song_name = body.get("song_name", "")
        artist_name = body.get("artist_name", "")
        lyrics = body.get("lyrics", "")
        cover_url = body.get("cover_url", "")
        device_id = body.get("device_id", "anonymous")
    else:
        form = await request.form()
        qq_music_url = form.get("url", "") or form.get("qq_music_url", "")
        song_name = form.get("song_name", "")
        artist_name = form.get("artist_name", "")
        lyrics = form.get("lyrics", "")
        cover_url = form.get("cover_url", "")
        device_id = form.get("device_id", "anonymous")

    if not song_name:
        raise HTTPException(400, "请提供歌曲名称")

    # ===== 去重：同一首歌只保留第一次结果 =====
    existing = _find_existing_review(song_name, artist_name)
    if existing:
        return existing

    review_id = str(uuid.uuid4())[:8]

    # 尝试获取音频并分析
    audio_features = {}
    songmid_for_audio = ""

    # 从 URL 提取 songmid 或 songid（用于获取音频）
    if qq_music_url:
        url_to_parse = qq_music_url.strip()
        # 处理短链接
        if 'y.qq.com' in url_to_parse:
            # 先尝试直接匹配
            has_direct = False
            for pat in [r'songDetail/([a-zA-Z0-9]+)', r'songmid=([a-zA-Z0-9]+)', r'songid=(\d+)']:
                if re.search(pat, url_to_parse):
                    has_direct = True
                    break
            if not has_direct:
                try:
                    req = urllib.request.Request(url_to_parse)
                    req.add_header('User-Agent', 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)')
                    resp = urllib.request.urlopen(req, timeout=8)
                    url_to_parse = resp.geturl()
                except Exception:
                    pass

        # 提取 songmid
        for pat in [r'songDetail/([a-zA-Z0-9]+)', r'songmid=([a-zA-Z0-9]+)', r'song/([a-zA-Z0-9]{10,})']:
            m = re.search(pat, url_to_parse)
            if m and not m.group(1).isdigit():
                songmid_for_audio = m.group(1)
                break

        # 如果没有 songmid，尝试从 songid 转换
        if not songmid_for_audio:
            songid_match = re.search(r'songid=(\d+)', url_to_parse)
            if songid_match:
                songid_val = songid_match.group(1)
                logger.info(f"No songmid found, converting songid={songid_val} to songmid...")
                try:
                    detail = await _fetch_song_detail(songid=songid_val)
                    songmid_for_audio = detail.get("songmid", "")
                    logger.info(f"Converted songid={songid_val} -> songmid={songmid_for_audio}")
                except Exception as e:
                    logger.warning(f"Failed to convert songid to songmid: {e}")

    # 尝试下载试听音频并分析
    if songmid_for_audio:
        try:
            audio_features = await _download_and_analyze_audio(songmid_for_audio)
            logger.info(f"Audio analysis from link successful: BPM={audio_features.get('bpm')}")
        except Exception as e:
            logger.warning(f"Audio analysis from link failed: {e}")
            # 无法获取音频时，基于歌词做深度 NLP 分析弥补
            if lyrics:
                audio_features = _analyze_lyrics_deeply(lyrics, song_name, artist_name)
                logger.info("Falling back to deep lyrics analysis")
    else:
        logger.info("No songmid available for audio analysis")
        if lyrics:
            audio_features = _analyze_lyrics_deeply(lyrics, song_name, artist_name)

    # Generate reviews
    ding_prompt = build_review_prompt(song_name, artist_name, lyrics, audio_features, "ding")
    liang_prompt = build_review_prompt(song_name, artist_name, lyrics, audio_features, "liang")

    ding_result = await call_llm(DING_SYSTEM_PROMPT, ding_prompt)
    liang_result = await call_llm(LIANG_SYSTEM_PROMPT, liang_prompt)

    # Save to DB
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        INSERT INTO reviews (id, device_id, song_name, artist_name, cover_url, input_mode,
            qq_music_url, lyrics, audio_features,
            ding_review, ding_scores, ding_total,
            liang_review, liang_scores, liang_total, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        review_id, device_id, song_name, artist_name, cover_url, "link",
        qq_music_url, lyrics, json.dumps(audio_features),
        ding_result["review"], json.dumps(ding_result["scores"]), ding_result["total"],
        liang_result["review"], json.dumps(liang_result["scores"]), liang_result["total"],
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

    return {
        "id": review_id,
        "song_name": song_name,
        "artist_name": artist_name,
        "cover_url": cover_url,
        "audio_features": audio_features,
        "ding": ding_result,
        "liang": liang_result,
    }


@app.get("/api/review/{review_id}")
async def get_review(review_id: str):
    """获取评测结果详情"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "评测记录不存在")

    return {
        "id": row["id"],
        "song_name": row["song_name"],
        "artist_name": row["artist_name"],
        "cover_url": row["cover_url"],
        "input_mode": row["input_mode"],
        "audio_features": json.loads(row["audio_features"]) if row["audio_features"] else {},
        "ding": {
            "review": row["ding_review"],
            "scores": json.loads(row["ding_scores"]) if row["ding_scores"] else {},
            "total": row["ding_total"],
        },
        "liang": {
            "review": row["liang_review"],
            "scores": json.loads(row["liang_scores"]) if row["liang_scores"] else {},
            "total": row["liang_total"],
        },
        "created_at": row["created_at"],
    }


@app.get("/api/reviews")
async def list_reviews(device_id: str = "anonymous", limit: int = 20):
    """获取历史评测列表"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, song_name, artist_name, cover_url, ding_total, liang_total, created_at FROM reviews WHERE device_id = ? ORDER BY created_at DESC LIMIT ?",
        (device_id, limit)
    ).fetchall()
    conn.close()

    return [dict(r) for r in rows]


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "2.1",
        "llm_mode": LLM_MODE,
        "hunyuan_configured": bool(HUNYUAN_SECRET_ID and HUNYUAN_SECRET_KEY),
        "model": os.environ.get("HUNYUAN_MODEL", "hunyuan-lite"),
    }


# ====== Main ======
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
