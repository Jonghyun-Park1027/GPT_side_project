import os
import pathlib
from datetime import datetime

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


# ───────────────────────────────────── Sidebar ───────────────────────────────────
st.sidebar.header("🤖 Cloudflare SiteGPT")
api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
st.sidebar.markdown(
    "[📂 GitHub Repository](https://github.com/your-github-id/cloudflare-sitegpt)"
)

if not api_key:
    st.info("먼저 OpenAI API Key를 입력하세요.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ───────────────────────────────────── Settings ──────────────────────────────────
SITEMAP_INDEX = "https://developers.cloudflare.com/sitemap-0.xml"
CF_PRODUCTS = {
    "AI Gateway": "ai-gateway",
    "Vectorize": "vectorize",
    "Workers AI": "workers-ai",
}
NOMAD_URL = "https://nomadcoders.co/c/gpt-challenge/lobby"

CACHE_DIR = pathlib.Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# ────────────────────────────────── Loader Functions ─────────────────────────────


def load_cf_docs(selected_slugs: list[str]):
    """Load Cloudflare docs from master sitemap filtered by product slugs"""
    patterns = [rf".*{slug}/.*" for slug in selected_slugs]
    loader = SitemapLoader(web_path=SITEMAP_INDEX, filter_urls=patterns)
    docs = loader.load()
    for d in docs:
        # Extract product name from URL path
        for name, slug in CF_PRODUCTS.items():
            if f"/{slug}/" in d.metadata.get("source", ""):
                d.metadata["product"] = name
                break
    return docs


def load_nomad_doc():
    loader = WebBaseLoader(NOMAD_URL)
    docs = loader.load()
    for d in docs:
        d.metadata["product"] = "Nomad GPT Challenge"
    return docs


def build_or_load_chroma(selected_products: list[str]):
    key = "_".join(sorted(selected_products)).replace(" ", "_").lower()
    store_path = CACHE_DIR / f"{key}_chroma"

    # Chroma uses a directory for persistence
    if store_path.exists() and any(store_path.iterdir()):
        return Chroma(
            persist_directory=str(store_path), embedding_function=OpenAIEmbeddings()
        )

    # --- Load docs ---
    cf_slugs = [CF_PRODUCTS[p] for p in selected_products if p in CF_PRODUCTS]
    docs = load_cf_docs(cf_slugs)
    if "Nomad GPT Challenge" in selected_products:
        docs += load_nomad_doc()

    # --- Split & embed ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=str(store_path)
    )
    vectordb.persist()
    return vectordb


# ───────────────────────────────────── UI Inputs ─────────────────────────────────
available_products = list(CF_PRODUCTS.keys()) + ["Nomad GPT Challenge"]
selected_products = st.sidebar.multiselect(
    "대상 문서", available_products, default=available_products
)


# ───────────────────────────────────── Retriever ─────────────────────────────────
@st.cache_resource(show_spinner=True, ttl=24 * 3600)
def get_retriever(products):
    vectordb = build_or_load_chroma(products)
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})


retriever = get_retriever(selected_products)

# ───────────────────────────────────── QA Chain ──────────────────────────────────
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, return_source_documents=True
)

# ───────────────────────────────────── Main Area ─────────────────────────────────
st.title("Cloudflare Docs GPT 🤖")
query = st.text_input("Cloudflare 공식 문서에 대해 질문하세요:")

if query:
    with st.spinner("답변 생성 중..."):
        result = qa_chain(query)
        st.subheader("📘 답변")
        st.write(result["result"])

        with st.expander("🔍 참고 문서"):
            for doc in result["source_documents"]:
                url = doc.metadata.get("source", "")
                title = doc.metadata.get("title", url)
                st.markdown(f"- [{title}]({url})")

st.markdown("---")
st.markdown("예시 질문:")
st.markdown("- llama-2-7b-chat-fp16 모델의 1M 입력 토큰당 가격은 얼마인가요?")
st.markdown("- Cloudflare의 AI 게이트웨이로 무엇을 할 수 있나요?")
st.markdown("- 벡터라이즈에서 단일 계정은 몇 개의 인덱스를 가질 수 있나요?")

st.caption(f"Index last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
