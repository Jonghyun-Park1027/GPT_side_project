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
gh_url = "https://github.com/Jonghyun-Park1027/GPT_side_project/tree/main"  # 수정하세요
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
def _generate_quiz_locally(difficulty: str, num_questions: int) -> list[dict]:
    client = get_client()
    prompt = (
        f"Create {num_questions} {difficulty} general-knowledge quiz questions.\n"
        "Return *only* valid JSON list. "
        "Each item must have keys: question, answer."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        content = resp.choices[0].message.content
        if not content:
            raise ValueError("응답이 비어 있습니다.")
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        try:
            quiz = json.loads(content)
        except Exception as e:
            st.error(
                "⚠️ 퀴즈 생성에 실패했습니다. 다시 시도해 주세요.\n\n"
                "에러: JSON 파싱 실패. 모델이 JSON 이외의 형식(코드블록 등)으로 응답했을 수 있습니다.\n"
                f"원본 응답:\n\n{content}\n\n에러: {e}"
            )
            st.stop()
        if not isinstance(quiz, list):
            raise ValueError("응답이 리스트 형태의 JSON이 아닙니다.")
    except Exception as e:
        st.error(f"⚠️ 퀴즈 생성에 실패했습니다. 다시 시도해 주세요.\n\n에러: {e}")
        st.stop()
    return quiz


# ---------- LLM 함수 호출 엔드포인트 ----------
def create_quiz_with_function_call(diff: str, n: int) -> list[dict]:
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
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            functions=FUNC_SPEC,  # type: ignore
            function_call={"name": "generate_quiz"},
        )
        choice = resp.choices[0]
        msg = choice.message
        function_call = getattr(msg, "function_call", None)
        if not function_call or not getattr(function_call, "arguments", None):
            raise ValueError(
                f"함수 호출 정보가 없습니다. (function_call: {function_call})\n"
                f"모델 응답: {getattr(msg, 'content', '') or msg}"
            )
        arguments = function_call.arguments
        args = json.loads(arguments)
        return _generate_quiz_locally(
            difficulty=args["difficulty"], num_questions=args["num_questions"]
        )
    except Exception as e:
        st.error(f"⚠️ 퀴즈 생성에 실패했습니다. 다시 시도해 주세요.\n\n에러: {e}")
        st.stop()


# ---------- 시험 시작 / 재시작 ----------
def start_new_quiz():
    st.session_state.quiz = create_quiz_with_function_call(difficulty, num_questions)
    st.session_state.answers = {}
    st.session_state.score = None
    st.session_state.quiz_ready = True


# ---------- UI ----------
st.title("📝 QuizGPT")

# Start Quiz 버튼을 누르면 바로 문제를 생성하고 화면에 표시
start_clicked = st.session_state.get("start_clicked", False)
if not st.session_state.quiz_ready and not start_clicked:
    st.write("설정을 확인한 뒤 **Start Quiz** 버튼을 눌러 주세요!")
    if st.button("🚀 Start Quiz"):
        start_new_quiz()
        st.session_state.start_clicked = True
        st.rerun()
    st.stop()
elif not st.session_state.quiz_ready and start_clicked:
    # 문제 생성 중이거나 생성 직후 rerun
    st.write("문제를 생성 중입니다. 잠시만 기다려 주세요...")
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
    st.session_state.quiz_ready = False
    st.session_state.start_clicked = False  # 결과 화면에서 다시 시작 가능

# ---------- 결과 ----------
if st.session_state.score is not None:
    total = len(quiz)
    score = st.session_state.score
    st.subheader(f"🎯 점수 : {score} / {total}")

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
            st.session_state.start_clicked = False
            st.rerun()
