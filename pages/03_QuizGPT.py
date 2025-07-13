import os, json
import streamlit as st
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessage,
)

# ---------- Sidebar : API Key · 깃허브 링크 · 옵션 ----------
st.sidebar.header("🎯 QuizGPT 설정")
api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
gh_url = "https://github.com/your-github-id/QuizGPT"  # 수정하세요
st.sidebar.markdown(f"[📂 GitHub 리포지터리]({gh_url})")

difficulty = st.sidebar.selectbox(
    "🧩 난이도",
    options=["easy", "hard"],
    index=0,
    help="easy → 쉬운 문제, hard → 어려운 문제",
)

num_questions = st.sidebar.slider(
    "❓ 문제 수", 3, 15, 5, help="한 회차에 출제될 문제 개수"
)

# ---------- 세션 상태 ----------
if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "score" not in st.session_state:
    st.session_state.score = None
if "quiz_ready" not in st.session_state:
    st.session_state.quiz_ready = False


# ---------- OpenAI 클라이언트 ----------
def get_client():
    if not api_key:
        st.warning("좌측 사이드바에 OpenAI API Key를 입력하세요.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI(api_key=api_key)


# ---------- LLM 함수 호출 스펙 ----------
FUNC_SPEC = [
    {
        "name": "generate_quiz",
        "description": "주어진 난이도와 개수에 맞게 퀴즈를 생성한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "hard"],
                    "description": "퀴즈 난이도",
                },
                "num_questions": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "문제 개수",
                },
            },
            "required": ["difficulty", "num_questions"],
        },
    }
]


# ---------- 실제 퀴즈 생성 함수 ----------
def _generate_quiz_locally(diff: str, n: int) -> list[dict]:
    """LLM에 JSON 형태로 퀴즈 자체를 생성하게 한다."""
    client = get_client()
    prompt = (
        f"Create {n} {diff} general-knowledge quiz questions.\n"
        "Return *only* valid JSON list. "
        "Each item must have keys: question, answer."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    # JSON 파싱
    content = resp.choices[0].message.content
    try:
        if content is None:
            raise ValueError("응답이 비어 있습니다.")
        quiz = json.loads(content)
    except Exception:
        st.error("⚠️ 퀴즈 생성에 실패했습니다. 다시 시도해 주세요.")
        st.stop()
    return quiz


# ---------- LLM 함수 호출 엔드포인트 ----------
def create_quiz_with_function_call(diff: str, n: int) -> list[dict]:
    """
    1) 모델이 generate_quiz 함수를 '호출'하도록 한다.
    2) arguments를 읽어 내부 _generate_quiz_locally 로직 실행.
    """
    client = get_client()
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are QuizGPT. "
            "When the user requests a quiz, you must call the "
            "function `generate_quiz` with proper arguments.",
        },
        {"role": "user", "content": f"Give me a {diff} quiz with {n} questions."},
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=FUNC_SPEC,  # type: ignore
        function_call={"name": "generate_quiz"},  # 함수 호출 강제
    )

    # finish_reason은 choices[0].finish_reason에 있음
    choice = resp.choices[0]
    msg = choice.message
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason != "function_call":
        st.error("함수 호출이 일어나지 않았습니다. 다시 시도해 주세요.")
        st.stop()

    # function_call이 None일 수 있으니 체크
    if not hasattr(msg, "function_call") or msg.function_call is None:
        st.error("함수 호출 정보가 없습니다. 다시 시도해 주세요.")
        st.stop()

    arguments = getattr(msg.function_call, "arguments", None)
    if not arguments:
        st.error("함수 호출 인자가 없습니다. 다시 시도해 주세요.")
        st.stop()

    try:
        args = json.loads(arguments)
    except Exception:
        st.error("함수 호출 인자 파싱에 실패했습니다. 다시 시도해 주세요.")
        st.stop()
    return _generate_quiz_locally(**args)


# ---------- 시험 시작 / 재시작 ----------
def start_new_quiz():
    st.session_state.quiz = create_quiz_with_function_call(difficulty, num_questions)
    st.session_state.answers = {}
    st.session_state.score = None
    st.session_state.quiz_ready = True


# ---------- UI ----------
st.title("📝 QuizGPT")

if not st.session_state.quiz_ready:
    st.write("설정을 확인한 뒤 **Start Quiz** 버튼을 눌러 주세요!")
    if st.button("🚀 Start Quiz"):
        start_new_quiz()
    st.stop()

quiz = st.session_state.quiz

# ---------- 문제 표시 ----------
with st.form("quiz_form"):
    for idx, q in enumerate(quiz, start=1):
        st.markdown(f"**Q{idx}. {q['question']}**")
        st.text_input("답 :", key=f"answer_{idx}")
    submitted = st.form_submit_button("✅ 제출")

# ---------- 채점 ----------
if submitted:
    answers = {}
    score = 0
    for idx, q in enumerate(quiz, start=1):
        user_ans = st.session_state.get(f"answer_{idx}", "").strip()
        correct_ans = q["answer"].strip()
        answers[idx] = {"user": user_ans, "correct": correct_ans}
        if user_ans.lower() == correct_ans.lower():
            score += 1

    st.session_state.answers = answers
    st.session_state.score = score
    st.session_state.quiz_ready = False  # 결과 화면 상태 전환

# ---------- 결과 ----------
if st.session_state.score is not None:
    total = len(quiz)
    score = st.session_state.score
    st.subheader(f"🎯 점수 : {score} / {total}")

    # 상세 정오표
    with st.expander("정답 확인"):
        for idx in range(1, total + 1):
            ua = st.session_state.answers[idx]["user"] or "🈳 (무응답)"
            ca = st.session_state.answers[idx]["correct"]
            emoji = "✅" if ua.lower() == ca.lower() else "❌"
            st.markdown(
                f"**Q{idx}** {emoji}  \n"
                f"- Your answer : {ua}  \n"
                f"- Correct      : {ca}"
            )

    if score == total:
        st.success("만점입니다! 축하합니다! 🎉")
        st.balloons()
    else:
        if st.button("🔄 다시 도전하기"):
            start_new_quiz()
