# app.py
# -------------------------------------------------------------
# Streamlit 기반 RAG-Lite Chat – 파일 업로드, 채팅 히스토리
# -------------------------------------------------------------
import os, tempfile, requests, streamlit as st
from openai import OpenAI

# ---------- 사이드바 ----------
st.sidebar.title("⚙️ 설정")
repo_link = "https://github.com/<your-org-or-id>/<repo>/blob/main/app.py"
st.sidebar.markdown(f"[📂 GitHub에서 소스 보기]({repo_link})")

st.title("🗂️ RAG-Lite Chat")

# ---------- API 클라이언트 ----------
# setx로 환경변수에 OPENAI_API_KEY를 미리 등록했으므로 별도 입력 없이 사용
client = OpenAI()

# ---------- 세션 상태 초기화 ----------
if "assistant_id" not in st.session_state:
    st.session_state.assistant_id = None
    st.session_state.thread_id = None
    st.session_state.vector_store_id = None
    st.session_state.chat_history = []  # [{role, content}, …]

# ---------- 파일 업로드 ----------
uploaded = st.sidebar.file_uploader(
    "📝 지식 파일 업로드 (.txt, .pdf, .docx 등)",
    type=["txt", "pdf", "docx"],
    help="문서를 올리면 자동으로 벡터스토어를 만들고 연결합니다.",
)


def bootstrap_assistant(file_path: str, file_name: str):
    """파일을 OpenAI에 올리고 Assistant/Thread/VectorStore를 한 번에 초기화"""
    # 1) 파일 업로드
    file_obj = client.files.create(file=open(file_path, "rb"), purpose="assistants")
    # 2) 벡터스토어 생성 + 파일 등록
    vs = client.vector_stores.create(name=f"rag_vs_{file_name}")
    client.vector_stores.files.create_and_poll(
        vector_store_id=vs.id, file_id=file_obj.id
    )
    # 3) Assistant 생성
    assistant = client.beta.assistants.create(
        name="RAG-Lite",
        instructions="근거 문장을 인용해 한국어로 간단히 답하세요.",
        model="gpt-4o-mini",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vs.id]}},
    )
    # 4) Thread 생성
    thread = client.beta.threads.create()

    # 세션 상태 저장
    st.session_state.assistant_id = assistant.id
    st.session_state.thread_id = thread.id
    st.session_state.vector_store_id = vs.id
    st.session_state.chat_history.clear()


# 처음 접속 시 파일 없으면 Gist 원문 기본 로딩(옵션)
DEFAULT_GIST = "https://gist.githubusercontent.com/serranoarevalo/5acf755c2b8d83f1707ef266b82ea223/raw/"
if st.session_state.assistant_id is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        txt = requests.get(DEFAULT_GIST, timeout=30).text
        tmp.write(txt.encode("utf-8"))
        tmp_path = tmp.name
    bootstrap_assistant(tmp_path, "aaronson_gist.txt")

# 새 파일 업로드 시 재초기화
if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp:
        tmp.write(uploaded.read())
        up_path = tmp.name
    bootstrap_assistant(up_path, uploaded.name)
    st.success("🔄 새 문서로 벡터스토어를 갱신했습니다!")

# ---------- 채팅 UI ----------
# 과거 대화 렌더링
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 입력창
user_query = st.chat_input("문서 내용에 대해 질문해 보세요…")
if user_query:
    # 1) 화면에 사용자 메시지 표시
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2) OpenAI Thread에 사용자 메시지 추가
    client.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=user_query,
    )

    # 3) Assistant 실행
    run = client.beta.threads.runs.create_and_poll(
        thread_id=st.session_state.thread_id,
        assistant_id=st.session_state.assistant_id,
    )

    # 4) 최신 Assistant 메시지 가져오기
    msgs = client.beta.threads.messages.list(thread_id=st.session_state.thread_id).data
    assistant_msgs = [m for m in msgs if m.role == "assistant"]
    if assistant_msgs:
        answer = assistant_msgs[0].content[0].text.value
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

# ---------- 하단 주석 ----------
st.markdown("---")
st.caption("ⓒ 2025 RAG-Lite Streamlit 데모")
