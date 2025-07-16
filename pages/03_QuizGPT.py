import json
import os
from httpx import Response
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core import output_parsers
from pydantic import Json
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser

#
# JSONDecodeError: Expecting value: line 1 column 1 (char 0) 에러는
# json.loads()가 빈 문자열이나 유효하지 않은 JSON을 파싱하려고 할 때 발생합니다.
# 즉, LLM의 출력이 비어있거나, JSON 형식이 아니거나, 앞에 불필요한 텍스트가 있을 때 주로 발생합니다.
#
# 원인 예시:
# - LLM이 아무 응답도 반환하지 않음 (빈 문자열)
# - LLM이 JSON이 아닌 텍스트(예: 설명, 코드블록 마크다운 등)를 반환
# - LLM이 JSON 앞에 불필요한 텍스트(예: "Here is your quiz:" 등)를 붙여서 반환
#
# 해결 방법:
# - LLM 프롬프트를 더 명확하게 하거나, 응답에서 JSON만 추출하는 후처리 추가
# - 출력이 비어있을 때 예외처리 추가


class JsonOutputParsor(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        text = text.strip()
        if not text:
            raise ValueError("LLM output is empty. Cannot parse JSON.")
        return json.loads(text)


output_parsers = JsonOutputParsor()
st.set_page_config(page_title="QuizGPT", page_icon="🤔")

st.title("QuizGPT")
llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
        """,
        )
    ]
)
questions_chain = {"context": format_docs} | questions_prompt | llm
formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }}
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm


@st.cache_resource(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Search WIkipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(wiki_client=None, top_k_results=5)

    docs = retriever.get_relevant_documents(term)
    return docs


@st.cache_data(show_spinner="Making quiz....")
def run_quiz_chain(_docs):
    chain = {"context": questions_chain} | formatting_chain | JsonOutputParsor()
    response = chain.invoke(_docs)
    return response


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use",
        (
            "File",
            "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt,or .pdf file", type=["pdf", "txt", "docx"]
        )
        if file:
            docs = split_file(file)
            st.write(docs)
    else:
        topic = st.text_input("Search WIkipedia...")

        if topic:
            docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
        Welcome to QuizGPT.
        I will make a quiz from wikipedia articles or files you upload to test your knowledge and help you study.
        Get started by uploading a file or searching on WIkipedia in the sidebar
        """
    )
else:
    if "quiz_generated" not in st.session_state:
        st.session_state.quiz_generated = False
        st.session_state.quiz_response = None

    if not st.session_state.quiz_generated:
        if st.button("generate quiz"):
            st.session_state.quiz_response = run_quiz_chain(docs)
            st.session_state.quiz_generated = True

    if st.session_state.quiz_generated and st.session_state.quiz_response:
        response = st.session_state.quiz_response
        st.write(response)
        with st.form("questions_form"):
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                )
                if st.write({"answer": value, "correct": True} in question["answers"]):
                    st.success("Correct")
                elif value is not None:
                    st.error("Wrong")
            button = st.form_submit_button()
