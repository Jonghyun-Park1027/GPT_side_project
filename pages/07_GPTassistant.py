# -----------------------------------------------
# app.py ― Security-Research Assistant (Streamlit)
# -----------------------------------------------
import os, json, time, textwrap, pathlib, re, requests
from datetime import datetime

import streamlit as st
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from openai import OpenAI, RateLimitError, APIConnectionError

st.set_page_config(
    page_title="GPT Assistant",
    page_icon="💼",
)

st.markdown(
    """
    # GPT Assistant
            
    Welcome to GPT Assistant.
            
    Write down the name of a company and our Agent will do the research for you.
"""
)
# ────────────────────────────────
# 1.  사이드바 : API Key + GitHub
# ────────────────────────────────
st.set_page_config(page_title="Security Research Assistant", page_icon="🔍")
with st.sidebar:
    st.header("🔑  OpenAI API Key")
    user_api_key = st.text_input("API Key를 입력하세요", type="password")
    st.markdown("---")
    st.header("📂  GitHub")
    st.markdown(
        "[프로젝트 소스 코드 보기](https://github.com/Jonghyun-Park1027)",  # ← 리포지토리 URL로 교체
        unsafe_allow_html=True,
    )

if not user_api_key:
    st.info("대화를 시작하려면 OpenAI API Key를 입력하세요.")
    st.stop()

client = OpenAI(api_key=user_api_key)

# ────────────────────────────────
# 2.  Session State 초기화
# ────────────────────────────────
if "assistant_id" not in st.session_state:
    assistant = client.beta.assistants.create(
        name="Security Research Assistant",
        model="gpt-4o-mini",  # 필요 시 gpt-4o
        instructions=textwrap.dedent(
            """
            You are a meticulous security researcher.
            When asked to investigate a topic, you should:
            1) Call `duckduckgo_search` for candidate sources.
            2) For each promising result, call `web_scrape` to pull the full text.
            3) Summarize findings into a concise report and send it back directly to the user.
            """
        ),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "duckduckgo_search",
                    "description": "Search DuckDuckGo and return the top 5 results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (e.g. 'xz backdoor exploit')",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_scrape",
                    "description": "Fetch plain-text content of a web page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "Target page URL"},
                        },
                        "required": ["url"],
                    },
                },
            },
        ],
    )
    st.session_state.assistant_id = assistant.id
    st.session_state.thread_id = client.beta.threads.create().id
    st.session_state.history = []  # Streamlit용 대화 이력

# ────────────────────────────────
# 3.  로컬 함수 구현 (Assistant → Tool Call)
# ────────────────────────────────


def duckduckgo_search(query: str) -> list[dict]:
    """Return DuckDuckGo top‑5 search results as a list of dicts."""
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
    return results


def web_scrape(url: str) -> str:
    """Scrape given URL and return cleaned plain‑text."""
    resp = requests.get(url, timeout=20)
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = re.sub(r"\n{2,}", "\n", soup.get_text("\n"))
    return text.strip()


LOCAL_TOOL_MAP = {
    "duckduckgo_search": duckduckgo_search,
    "web_scrape": web_scrape,
}

# ────────────────────────────────
# 4.  대화 UI - 기존 기록 출력
# ────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ────────────────────────────────
# 5.  사용자 입력 → Assistant 실행
# ────────────────────────────────
user_input = st.chat_input("검색·조사할 주제를 입력하세요")
if user_input:
    # (1) 화면에 즉시 표시
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # (2) 메시지를 Thread 에 추가
    client.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=user_input,
    )

    # (3) Assistant 실행
    run = client.beta.threads.runs.create(
        thread_id=st.session_state.thread_id,
        assistant_id=st.session_state.assistant_id,
    )

    # (4) Tool 처리 및 완료 대기
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=st.session_state.thread_id, run_id=run.id
        )

        # ── 4‑A. Tool 호출 요구 ─────────────────────────────
        if run.status == "requires_action":
            tool_outputs = []
            for call in run.required_action.submit_tool_outputs.tool_calls:
                fn_name = call.function.name
                args = json.loads(call.function.arguments)
                try:
                    result = LOCAL_TOOL_MAP[fn_name](**args)
                except Exception as e:
                    result = f"Tool execution error: {e}"
                tool_outputs.append(
                    {
                        "tool_call_id": call.id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )

            # Assistant 에 결과 제출
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=st.session_state.thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )
            continue

        # ── 4‑B. 완료 상태 ────────────────────────────────
        if run.status == "completed":
            messages = client.beta.threads.messages.list(
                thread_id=st.session_state.thread_id, order="asc"
            )
            # 최신 Assistant 메시지만 가져오기
            assistant_msg = next(
                (m for m in reversed(messages.data) if m.role == "assistant"), None
            )
            if assistant_msg:
                content = assistant_msg.content[0].text.value
                st.session_state.history.append(
                    {"role": "assistant", "content": content}
                )
                with st.chat_message("assistant"):
                    st.markdown(content)
            break

        # ── 4‑C. 오류 or 진행 중 ──────────────────────────
        if run.status in {"failed", "cancelled", "expired"}:
            st.error(f"Run ended with status: {run.status}")
            break
        time.sleep(1)  # 진행 중이면 잠시 대기
