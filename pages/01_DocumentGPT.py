from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from langchain_openai import ChatOpenAI
import os
from langchain.callbacks.base import BaseCallbackHandler


class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "AI")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(streaming=True, callbacks=[ChatCallbackHandler()])
st.title("Document GPT")

st.markdown(
    """
Welcome!

Use this chatbotto ask question to an AI about your files!

upload your file on sidebar
"""
)

with st.sidebar:
    file = st.file_uploader(
        "uploade a txt pdf or docx file", type=["pdf", "txt", "docx"]
    )

# 오류 검토:
# 1. st.session_state["messages"]가 항상 존재한다고 가정하고 있음. 초기화 필요.
# 2. embed_file 함수에서 항상 "./.cache/files/chapter_one.txt"만 로드함. 업로드한 파일을 사용해야 함.
# 3. chain이 정의만 되고 실행되지 않음.
# 4. "Use this chatbotto ask question..." 오타 있음.
# 5. "uploade a txt pdf or docx file" 오타 있음.

if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    # 오류: 항상 chapter_one.txt만 로드함. 아래처럼 수정 필요:
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = Chroma.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use the following portion of a long document to see if any of the text is relevant to answer the question. Return any relevant text verbatim. If there is no relevant text, return : ''
            -------
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

if file:
    retriever = embed_file(file)

    # Display chat history
    paint_history()

    # Show initial message only if no previous messages
    if not st.session_state["messages"]:
        send_message("I'm ready! ask away", "AI", save=True)

    message = st.chat_input("Ask anything about your file....")

    if message:
        send_message(message, "human")
        # 오류: chain이 실행되지 않음. 아래처럼 실행 필요.
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
        # send_message(response.content, "AI")

else:
    st.session_state["messages"] = []
