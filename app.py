import streamlit as st
# Get API key
api_key = st.secrets.get("GOOGLE_API_KEY")

from moviepy import VideoFileClip
import whisper
import subprocess
import os
from datetime import timedelta
import srt
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny")

import spacy

try:
    nlp = spacy.load("xx_sent_ud_sm")
except OSError:
    nlp = None



from pydantic import BaseModel, Field, ValidationError
from langchain_classic.docstore.document import Document
from langchain_classic.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
#---------database
import sqlite3
from datetime import datetime



#----------for translation
from argostranslate import translate, package

                # -----------------------------
                # Conversational Memory (Streamlit)
               # -----------------------------
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
#-------------databse initialization
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


#-----------------------------
# history builder
#-----------------------------
def build_history():
    history_text = ""
    for turn in st.session_state.conversation_memory[-5:]:
        history_text += f"User: {turn['user']}\n"
        history_text += f"AI: {turn['ai']}\n"
    return history_text


# -----------------------------
# Pydantic Models
# -----------------------------
class QAResponse(BaseModel):
    topic_in_video: str = Field(..., description="Whether topic is discussed in video")
    video_content: str = Field(..., description="Content from video if discussed, else N/A")
    general_answer: str = Field(..., description="General explanation/definition of the topic")
    confidence: float = Field(..., description="Confidence score (dummy or real)")

class SummaryResponse(BaseModel):
    summary: str = Field(..., description="Summary of video")
    confidence: float = Field(..., description="Confidence score (dummy or real)")

# -----------------------------
# Audio extraction
# -----------------------------
def extract_audio(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    if audio:
        audio.write_audiofile(output_audio_path)
        return True
    return False

#------------------------
#   translation
#----------------------
@st.cache_resource
def load_argos(from_code, to_code):
    package.update_package_index()
    available = package.get_available_packages()
    pkg = next(p for p in available if p.from_code == from_code and p.to_code == to_code)
    package.install_from_path(pkg.download())

# def translate_text(text, src_lang, tgt_lang):
#     if src_lang == tgt_lang:
#         return text
#     load_argos(src_lang, tgt_lang)
#     return translate.translate(text, src_lang, tgt_lang)
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





# -----------------------------
# Transcription using Whisper
# -----------------------------
# def transcribe(audio_path):
#     # def transcribe(audio_path):
#     st.write("⏳ Loading Whisper model...")
#     model = whisper.load_model("tiny")  # TEMP: use tiny for speed
#     st.write("🎧 Transcribing audio...")
#     result = model.transcribe(audio_path, verbose=True)
#     return result

#     # model = whisper.load_model("base")
#     # return model.transcribe(audio_path)
def transcribe(audio_path):
    st.write("⏳ Whisper model already loaded")
    model = load_whisper_model()
    st.write("🎧 Transcribing audio...")
    return model.transcribe(audio_path,task="transcribe", verbose=True)



# -----------------------------
# Generate SRT
# -----------------------------
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

# -----------------------------
# Burn subtitles
# -----------------------------
def burn_subtitles(video_path, srt_path, output_path):
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", f"subtitles={srt_path}",
        "-c:a", "copy",
        output_path
    ]
    
    subprocess.run(command, shell=True, check=True)

# -----------------------------
# Prompt Template for Chatbot
# -----------------------------
# video_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are a video-aware assistant.

# Rules:
# 1. First line: Say whether the topic is discussed in the video: "This topic is discussed in the video." or "This topic is NOT discussed in the video."
# 2. Second line: If discussed, summarize what the video says about it. If not discussed, write "N/A".
# 3. Third line: Provide a general explanation or definition of the topic.

# Video Transcript:
# {context}

# Question: {question}
# """
# )
video_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a strict, professional AI assistant that answers questions ONLY using the provided video transcript.



Follow these rules without exception:

1️⃣ **Strict Grounding**
- Use ONLY the provided transcript as your source.
- Do NOT use outside knowledge, memory, assumptions, or guesses.
- Do NOT invent facts, dates, names, or explanations.

2️⃣ **When Information Is Missing**
- If the transcript does NOT contain the answer, clearly state:
  "This topic is NOT discussed in the video."
- Do NOT provide any general knowledge, definitions, or external explanations.
- Output "N/A" for the transcript-based answer.

3️⃣ **Time-Sensitive or Current Questions**
- If the question requires current information (e.g., today’s date, current events, Islamic date, current officials):
  - Treat it as NOT discussed unless explicitly mentioned in the transcript.
  - Do NOT attempt to calculate or guess.
  - Do NOT reference websites or external sources.

4️⃣ **Output Format (EXACT)**
Return exactly three lines:

Line 1: Either  
"This topic is discussed in the video."  
OR  
"This topic is NOT discussed in the video."

Line 2:
- Transcript-based answer if present  
- Otherwise: "N/A"

Line 3:
- If discussed: a concise summary strictly from the transcript
- If not discussed: Provide a general explanation or definition of the topic.

5️⃣ **Tone**
- Professional, concise, factual
- No filler text
- No speculation

**Video Transcript:**
{context}

**Question:** {question}
"""
)

# -----------------------------
# Category Prompt (ADD HERE)
# -----------------------------
category_prompt_text = """
Classify the main topic of this video using ONLY the transcript.

Rules:
- Do NOT guess.
- Do NOT use external knowledge.
- Choose ONE best category.

Allowed Categories:
Education, Technology, Programming, Artificial Intelligence, Data Science,
Business, Entrepreneurship, Marketing, Finance, Health, Science,
Religion, Motivation, Tutorial / How-To, News, Entertainment, Other

Return EXACTLY in this format:

Category: <category> ,\n  
Reason: <give a valid reason for that.a one line from overall summary>
"""

# -----------------------------
# Create Video Chatbot Chain
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# def create_video_chatbot(transcript_text, api_key):
#     docs = [Document(page_content=transcript_text)]
#     splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
#     docs = splitter.split_documents(docs)
#     embeddings = load_embeddings()
#     vectorstore = Chroma.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     collection_name="video_transcript",
#     persist_directory=None
#    )
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, collection_name="video_transcript")
    # retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=ChatGoogleGenerativeAI(
    #         model="gemini-2.5-flash",
    #         api_key=api_key,
    #         temperature=0
    #     ),
    #     retriever=retriever,
    #     return_source_documents=True,
    #     chain_type="stuff",
    #     chain_type_kwargs={"prompt": video_prompt}
    # )
    # return qa_chain

def create_video_chatbot(transcript_text, api_key):
    docs = [Document(page_content=transcript_text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    docs = splitter.split_documents(docs)

    embeddings = load_embeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="video_transcript"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4, "include_metadata": True})

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key,
            temperature=0
        ),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": video_prompt},
        return_source_documents=True
    )

    return qa_chain

def calculate_confidence(result):
    """
    Calculates confidence based on retrieval relevance
    """
    source_docs = result.get("source_documents", [])

    # If nothing retrieved → very low confidence
    if not source_docs:
        return 0.15

    # Similarity proxy: more chunks = more confidence
    retrieval_score = min(len(source_docs) / 4, 1.0)

    # Transcript coverage
    total_chars = sum(len(doc.page_content) for doc in source_docs)
    coverage_score = min(total_chars / 1500, 1.0)

    # Detect NOT discussed
    answer = result["result"].lower()
    discussed_flag = 0.0 if "not discussed" in answer else 1.0

    confidence = (
        0.5 * retrieval_score
        + 0.3 * coverage_score
        + 0.2 * discussed_flag
    )

    return round(confidence, 2)


# -----------------------------
# Video Analyzer
# -----------------------------
def analyze_transcript(text, top_n=10):
    words = re.findall(r'\b\w+\b', text.lower())
    counter = Counter(words)
    most_common = counter.most_common(top_n)
    return most_common, counter

# def plot_wordcloud(text):
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
#     plt.figure(figsize=(10,5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     st.pyplot(plt)
def plot_wordcloud(text):
    if not text or not text.strip():
        st.warning("No text available for word cloud.")
        return

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc.to_array(), interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="AI Video Assistant", layout="wide")
st.title("📹 AI-powered Video Assistant")



# Upload video
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov"])
if uploaded_video:
    os.makedirs("videos", exist_ok=True)
    video_path = "videos/uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    st.video(video_path)

    # st.video(video_path)

    # Transcription
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
      
    #Selection
    st.subheader("Caption Language Selection")

    target_language = st.selectbox(
    "Select language for captions",
    list(LANG_MAP.keys())
    )

    # Captions
    st.subheader("Generate transcript")
    if st.button("Generate Transcript") and st.session_state["transcript_text"]:
        #-----------translate before srt
        # src_lang = transcription_result["language"]
        src_lang = st.session_state["transcription_result"]["language"]

        target_code = LANG_MAP[target_language]

        # Translate full text
        translated_text = translate_text(
        st.session_state["transcription_result"]["text"],
        src_lang,
        target_code
        )

        # Translate segments
        for seg in st.session_state["transcription_result"]["segments"]:
           seg["text"] = translate_text(
           seg["text"],
           src_lang,
           target_code
   )


        st.write("Detected language:", src_lang)
        st.write("Target language:", target_code)
        # st.write(st.session_state["transcription_result"]["segments"][0]["text"])
        translated_full_text = "\n".join(
            seg["text"] for seg in st.session_state["transcription_result"]["segments"]
        )

        st.subheader("📜 Full Translated Transcript")
        st.text_area(
             "Translated Transcript",
             translated_full_text,
             height=300
            )



        # srt_content = convert_to_srt(transcription_result)
        srt_content = convert_to_srt(st.session_state["transcription_result"])

        os.makedirs("captions", exist_ok=True)
        srt_path = "captions/output.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
       
        st.info("Burning subtitles...")
        burn_subtitles(video_path, srt_path, "videos/output_video.mp4")
        st.success("✅ Video with captions ready!")
        st.video("videos/output_video.mp4")
        

    # -----------------------------
    # Chatbot
    st.subheader("💬 Video Chatbot")
    user_input = st.text_input("You:", key="chat_input")
    if user_input:
        if st.session_state["transcript_text"]:
            try:
                chatbot = create_video_chatbot(st.session_state["transcript_text"], api_key)
                # result = chatbot.invoke(user_input)
                MEMORY_TRIGGERS = [
                     "previous question",
                     "last question",
                     "what did i ask",
                     "earlier question",
                     "before this"
                    ]

                def is_memory_question(text):
                   text = text.lower()
                   return any(t in text for t in MEMORY_TRIGGERS)
                if is_memory_question(user_input):
                   if len(st.session_state.conversation_memory) >= 1:
                      last = st.session_state.conversation_memory[-1]["user"]
                      st.success(f"Your previous question was:\n\n**{last}**")
            


                result = chatbot.invoke({"query": user_input})
                
                confidence = calculate_confidence(result)
                
                answer_text = result["result"]
                
                qa_validated = QAResponse(
                    topic_in_video=answer_text.splitlines()[0],
                    video_content=answer_text.splitlines()[1],
                    general_answer=answer_text.splitlines()[2],
                    confidence=confidence
                )
                
                video_content_clean = qa_validated.video_content.strip().upper()
                
                if video_content_clean == "N/A":
                    response_text = f"""
                **Present in Video:** {qa_validated.topic_in_video}
                
                **General Answer:**  
                {qa_validated.general_answer}
                """
                else:
                    response_text = f"""
                **Present in Video:** {qa_validated.topic_in_video}
                
                **Answer from Video:**  
                {qa_validated.video_content}
                
                **General Answer:**  
                {qa_validated.general_answer}
                """
                
                st.markdown("### 🤖 Chatbot Response")
                st.markdown(response_text)
                st.write("Confidence:", qa_validated.confidence)
                
                st.session_state.conversation_memory.append({
                    "user": user_input,
                    "ai": response_text
                })
                
                log_interaction(
                    task_type="QA",
                    user_query=user_input,
                    agent_response=response_text,
                    confidence=qa_validated.confidence
                )

                # Render chat
                for sender, msg in st.session_state.chat_history:
                    if sender == "You":
                        st.markdown(f"**You:** {msg}")
                    else:
                        st.markdown(f"**Chatbot:** {msg}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Transcript not ready yet!")

    # -----------------------------
    # Summarization
    st.subheader("📝 Video Summarization")
    if st.button("Summarize Video"):
        if st.session_state["transcript_text"]:
            try:
                chatbot = create_video_chatbot(st.session_state["transcript_text"], api_key)
                summary_result = chatbot.invoke("Summarize this video transcript in a few sentences.")
                summary_validated = SummaryResponse(summary=summary_result["result"], confidence=confidence)
                st.write(summary_validated.summary)
                st.write("Confidence:", summary_validated.confidence)
                log_interaction(
                    task_type="SUMMARY",
                    user_query="Summarize video",
                    agent_response=summary_validated.summary,
                    confidence=summary_validated.confidence
)

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Transcript not ready yet!")

    # -----------------------------
    # Video Categorization
    # -----------------------------
    st.subheader("🏷️ Video Categorization")

    if st.button("Categorize Video"):
        if st.session_state["transcript_text"]:
            try:
                chatbot = create_video_chatbot(
                    st.session_state["transcript_text"],
                    api_key
                )

                category_result = chatbot.invoke(
                    category_prompt_text
                )

                st.write(category_result["result"])
                log_interaction(
                   task_type="CATEGORY",
                   user_query="Categorize video",
                   agent_response=category_result["result"],
                   confidence=confidence
                )



            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Transcript not ready yet!")


    # -----------------------------
    # Video Analyzer
    st.subheader("📊 Video Transcript Analysis")
    if st.session_state["transcript_text"]:
        most_common, counter = analyze_transcript(st.session_state["transcript_text"], top_n=15)
        # st.markdown("**Top Keywords in Video:**")
        # for word, count in most_common:
            # st.write(f"{word}: {count}")
        col1, col2 = st.columns([1,2])

        with col1:
          st.markdown("### 🔑 Top Keywords")
          for word, count in most_common:
            st.write(f"{word} ({count})")

        with col2:
          st.markdown("### ☁️ Word Cloud")
          plot_wordcloud(st.session_state["transcript_text"])


        # st.markdown("**Word Cloud:**")
        # plot_wordcloud(st.session_state["transcript_text"])

    st.subheader("🗄️ Agent Interaction Logs")

    if st.checkbox("Show Database Logs"):
      conn = sqlite3.connect("agent_logs.db")
      cursor = conn.cursor()

      cursor.execute("SELECT * FROM interactions ORDER BY id DESC")
      rows = cursor.fetchall()
      conn.close()

      for row in rows:
        st.markdown(f"""
        **ID:** {row[0]}  
        **Time:** {row[1]}  
        **Task:** {row[2]}  
        **Query:** {row[3]}  
        **Response:** {row[4][:300]}  
        **Confidence:** {row[5]}
         ---
     """)






