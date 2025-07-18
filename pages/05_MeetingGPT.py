from langchain.storage import LocalFileStore
from langchain_community.vectorstores import Chroma
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
from langchain.document_loaders import TextLoader, UnstructuredFileLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings

has_transcript = os.path.exists("./.cache/podcast.txt")
# .env 파일에서 환경변수 불러오기
load_dotenv()

# 환경변수에서 API 키 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    temperature=0.1,
)
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlab=100,
)


@st.cache_data()
def embed_file(file_path):

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    # 오류: 항상 chapter_one.txt만 로드함. 아래처럼 수정 필요:
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = Chroma.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


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

            loader = TextLoader("./.cache/transcript.txt")

            docs = loader.load_and_split(text_splitter=splitter)
            # st.write(docs)
            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a consice summary of the following : "{text}"
                CONCISE SUMMARY:

                """
            )
            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            summary = first_summary_chain.invoke({"text": docs[0].page_content})

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary.
                """
            )
            refine_chain = refine_prompt | llm | StrOutputParser()
            with st.status("Summurizing...") as status:
                for num, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {num+1}/{len(docs)-1}")
                    summary = refine_chain.invoke(
                        {
                            "existing_summary": summary,
                            "context": doc.page_content,
                        }
                    )
                    st.write(summary)
            st.write(summary)

    with qa_tab:
        retriever = embed_file(transcript_path)
        docs = retriever.invoke("do they talk about marcus aurelius?")
        st.write(docs)
