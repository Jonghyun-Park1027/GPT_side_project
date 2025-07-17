import streamlit as st
import subprocess
from pydub import AudioSegment
import math
import glob
from dotenv import load_dotenv
import openai
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
has_transcript = os.path.exists("./.cache/podcast.txt")
# .env 파일에서 환경변수 불러오기
load_dotenv()

# 환경변수에서 API 키 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    temperature= 0.1,

)


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    final_transcript = ""
    for file in files:
        with open(file, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1", file=audio_file, language="ko"
            )
            final_transcript += transcript.text
    with open(destination, "w") as file:
        file.write(final_transcript)


st.title(
    """
    # MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])

    if video:
        chunks_folder = "./.cache/chunks"
        with st.status("Loading video...") as status:
            video_cotent = video.read()
            video_path = f"./.cache/{video.name}"
            audio_path = video_path.replace("mp4", "mp3")
            transcript_path = video_path.replace("mp4", "txt")
            with open(video_path, "rb") as f:
                f.write(video_cotent)
            status.update(label="Extracting audio...")
            extract_audio_from_video(video_path)
            status.update(label="Cutting Audio...")

            cut_audio_in_chunks(audio_path, 10, chunks_folder)
            status.update(label="Transcaribing audio...")
            transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(["Transcript", "Summary", "Q&A"])

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())

    with summary_tab:
        start = st.button("Generate summary")
        if start:
            
