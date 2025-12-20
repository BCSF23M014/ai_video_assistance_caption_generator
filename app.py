import sys
print(sys.executable)
import pkg_resources
print([pkg.key for pkg in pkg_resources.working_set])


import streamlit as st
# Get API key
api_key = st.secrets.get("GOOGLE_API_KEY")
print(api_key)
from moviepy import VideoFileClip
import whisper
import subprocess
import os
from datetime import timedelta


print("all are imported")


import srt
import spacy
try:
    nlp = spacy.load("xx_sent_ud_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("xx_sent_ud_sm")
    nlp = spacy.load("xx_sent_ud_sm")

print("spacy")

# -------------------------
# Whisper model loading
# -------------------------
@st.cache_resource
def load_whisper_model():
    # Ensure model path is explicit to avoid errors in Streamlit Cloud
    return whisper.load_model("tiny")

from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

from pydantic import BaseModel, Field
from langchain_classic.docstore.document import Document
from langchain_classic.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import sqlite3
from datetime import datetime
from argostranslate import translate, package

# -------------------------
# Conversational Memory
# -------------------------
if "conversation_memory" not in st.session_state:
   st.session_state.conversation_memory = []

LANG_MAP = {
    "English": "en",
    "Urdu": "ur",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Chinese": "zh",
    "Japanese": "ja",
    "Portuguese": "pt",
    "Russian": "ru",
    "Turkish": "tr",
    "Korean": "ko"
}

# -------------------------
# Database init
# -------------------------
def init_db():
    conn = sqlite3.connect("agent_logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            task_type TEXT NOT NULL,
            user_query TEXT,
            agent_response TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_interaction(task_type, user_query, agent_response, confidence):
    conn = sqlite3.connect("agent_logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO interactions (
            timestamp,
            task_type,
            user_query,
            agent_response,
            confidence
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        task_type,
        user_query,
        agent_response,
        confidence
    ))
    conn.commit()
    conn.close()

# -------------------------
# Build history
# -------------------------
def build_history():
    history_text = ""
    for turn in st.session_state.conversation_memory[-5:]:
        history_text += f"User: {turn['user']}\n"
        history_text += f"AI: {turn['ai']}\n"
    return history_text

# -------------------------
# Pydantic Models
# -------------------------
class QAResponse(BaseModel):
    topic_in_video: str = Field(..., description="Whether topic is discussed in video")
    video_content: str = Field(..., description="Content from video if discussed, else N/A")
    general_answer: str = Field(..., description="General explanation/definition of the topic")
    confidence: float = Field(..., description="Confidence score (dummy or real)")

class SummaryResponse(BaseModel):
    summary: str = Field(..., description="Summary of video")
    confidence: float = Field(..., description="Confidence score (dummy or real)")

# -------------------------
# Audio extraction
# -------------------------
def extract_audio(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    if audio:
        audio.write_audiofile(output_audio_path)
        return True
    return False

# -------------------------
# Translation
# -------------------------
@st.cache_resource
def load_argos(from_code, to_code):
    package.update_package_index()
    available = package.get_available_packages()
    pkg = next(p for p in available if p.from_code == from_code and p.to_code == to_code)
    package.install_from_path(pkg.download())

def translate_text(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text

    available = package.get_available_packages()
    direct = [p for p in available if p.from_code == src_lang and p.to_code == tgt_lang]

    if direct:
        package.install_from_path(direct[0].download())
        return translate.translate(text, src_lang, tgt_lang)

    # fallback via English
    if src_lang != "en" and tgt_lang != "en":
        mid = translate_text(text, src_lang, "en")
        return translate_text(mid, "en", tgt_lang)

    return text

# -------------------------
# Transcription using Whisper
# -------------------------
def transcribe(audio_path):
    st.write("⏳ Whisper model already loaded")
    model = load_whisper_model()
    st.write("🎧 Transcribing audio...")
    return model.transcribe(audio_path, task="transcribe", verbose=True)

# -------------------------
# Generate SRT
# -------------------------
def convert_to_srt(transcription_result):
    segments = transcription_result.get("segments", [])
    subtitles = []
    for i, seg in enumerate(segments):
        subtitles.append(srt.Subtitle(
            index=i+1,
            start=timedelta(seconds=seg["start"]),
            end=timedelta(seconds=seg["end"]),
            content=seg["text"].strip()
        ))
    return srt.compose(subtitles)

# -------------------------
# Burn subtitles
# -------------------------
def burn_subtitles(video_path, srt_path, output_path):
    """
    Correct FFmpeg path usage:
    - Use quotes to avoid spaces in paths
    - Works in Streamlit Cloud
    """
    
    # command = [ffmpeg_path, "-i", video_path, "-vf", f"subtitles={srt_path}", output_path]
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", f"subtitles={srt_path}",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(command, check=True)
    

# -------------------------
# Wordcloud + analysis
# -------------------------
def analyze_transcript(text, top_n=10):
    words = re.findall(r'\b\w+\b', text.lower())
    counter = Counter(words)
    most_common = counter.most_common(top_n)
    return most_common, counter

def plot_wordcloud(text):
    if not text or not text.strip():
        st.warning("No text available for word cloud.")
        return
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc.to_array(), interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="AI Video Assistant", layout="wide")
st.title("📹 AI-powered Video Assistant")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov"])
if uploaded_video:
    os.makedirs("videos", exist_ok=True)
    video_path = "videos/uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    st.video(video_path)

    if "transcript_text" not in st.session_state:
        st.session_state["transcript_text"] = ""
        st.session_state["chat_history"] = []
        os.makedirs("audio", exist_ok=True)
        audio_path = "audio/uploaded_audio.wav"
        st.info("Extracting audio and transcribing video... This may take a few minutes.")

        if extract_audio(video_path, audio_path):
            with st.spinner("🧠 Whisper is transcribing... Please wait"):
                transcription_result = transcribe(audio_path)
                st.session_state["transcription_result"] = transcription_result
                st.session_state["transcript_text"] = transcription_result.get("text", "")
                st.success("✅ Transcription completed!")
        else:
            st.error("❌ No audio found in video.")

    st.subheader("Caption Language Selection")
    target_language = st.selectbox("Select language for captions", list(LANG_MAP.keys()))

    st.subheader("Generate transcript")
    if st.button("Generate Transcript") and st.session_state["transcript_text"]:
        src_lang = st.session_state["transcription_result"]["language"]
        target_code = LANG_MAP[target_language]

        translated_text = translate_text(st.session_state["transcription_result"]["text"], src_lang, target_code)

        for seg in st.session_state["transcription_result"]["segments"]:
            seg["text"] = translate_text(seg["text"], src_lang, target_code)

        translated_full_text = "\n".join(seg["text"] for seg in st.session_state["transcription_result"]["segments"])

        st.subheader("📜 Full Translated Transcript")
        st.text_area("Translated Transcript", translated_full_text, height=300)

        srt_content = convert_to_srt(st.session_state["transcription_result"])
        os.makedirs("captions", exist_ok=True)
        srt_path = "captions/output.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        st.info("Burning subtitles...")
        burn_subtitles(video_path, srt_path, "videos/output_video.mp4")
        st.success("✅ Video with captions ready!")
        st.video("videos/output_video.mp4")


