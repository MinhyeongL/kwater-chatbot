from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote import logging
from prompts import Prompt

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data import DBManager
from agent_config import AgentConfig

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers import StrOutputParser



# API KEY 정보로드
load_dotenv()

# LangSmith 로깅
logging.langsmith("KwaterGPT")

# 페이지 설정
st.set_page_config(page_title="KwaterGPT", page_icon="💧")

# 페이지 제목
st.title("수자원 GPT")

# 번수 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = {}


# 이전 대화 출력
def print_message():
    for chat_message in st.session_state.messages:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


with st.sidebar:
    pass


# 문서 포맷팅
def format_doc(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# 3. 출력 파서 - 테이블 이름 추출 함수
def extract_tables(llm_response):
    """LLM 응답에서 테이블 이름 목록을 추출합니다."""

    # 1. 사용 가능한 테이블 정의
    AVAILABLE_TABLES = {
        "TB_C_RT": "실시간 측정 데이터 테이블. 센서의 실시간 측정값을 저장합니다.",
        "TB_AI_C_RT": "AI 분석 결과 테이블. AI가 분석한 예측값과 결과를 저장합니다.",
        "TB_AI_C_CTR": "AI 제어 결과 테이블. AI의 제어 명령과 결과를 저장합니다.",
        "TB_TAG_MNG": "태그 관리 테이블. 시스템에서 사용하는 태그(센서 등)의 메타데이터를 저장합니다."
    }

    tables = []
    for line in llm_response.split('\n'):
        if line.startswith("테이블:"):
            # 대괄호 안의 내용 추출
            content = line.replace("테이블:", "").strip()
            # 쉼표로 구분된 항목을 분리
            if '[' in content and ']' in content:
                content = content.replace('[', '').replace(']', '')

            tables = [table.strip() for table in content.split(',')]
            break

    # 유효한 테이블만 필터링
    valid_tables = [table for table in tables if table in AVAILABLE_TABLES]
    valid_tables.append("TB_TAG_MNG") if "TB_TAG_MNG" not in valid_tables else None
    return valid_tables


def db_research_agent(question: str):
    """
    DB 연구 에이전트
    """
    conf = AgentConfig(
        location_code="A",
        plant_code="SN",
        algorithm_code="C"
    )
    dbm = DBManager(conf)
    db = dbm.get_db_connection()
    prompt = Prompt()

    # 2. 테이블 선택 프롬프트 템플릿 작성
    table_selection_prompt = prompt.table_selection_prompt()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    select_table_chain = table_selection_prompt | llm | StrOutputParser()

    response = select_table_chain.invoke({"question": question})
    selected_tables = extract_tables(response)
    
    TABLE_SCHEMAS = {
        table: db.get_table_info([table]) for table in selected_tables
    }

    prompt = Prompt()
    sql_query_gen_prompt = prompt.sql_query_gen_prompt(TABLE_SCHEMAS)
    sql_query_gen_chain = sql_query_gen_prompt | llm | StrOutputParser()

    query = sql_query_gen_chain.invoke({"question": question})

    result_df = dbm.select_from_table(query)

    python_repl = PythonAstREPLTool(
            globals={"pd": pd, "plt": plt, "np": np, "sns": sns, "df": result_df.copy()},  # 필요한 모듈들을 globals에 추가
            locals=None
            # locals={"df": dataframe}  # 데이터프레임을 locals에 추가
        )
    
    agent = create_pandas_dataframe_agent(
            llm,
            result_df,
            verbose=False,
            agent_type="tool-calling",
            allow_dangerous_code=True,
            extra_tools=[python_repl],
            prefix="You are a professional data analyst and expert in Pandas. "
            "You must use Pandas DataFrame(`df`) to answer user's request. "
            "\n\n[IMPORTANT] DO NOT create or overwrite the `df` variable in your code. \n\n"
            "You don't need to generate visualization code."
            "But if you are willing to generate visualization code, please use `plt.show()` at the end of your code. "
            "I prefer seaborn code for visualization, but you can use matplotlib as well."
            "\n\n<Visualization Preference>\n"
            "- [IMPORTANT] Use `English` for your visualization title and labels."
            "The language of final answer should be written in Korean. "
            "\n\n###\n\n<Column Guidelines>\n"
            "If user asks with columns that are not listed in `df.columns`, you may refer to the most similar columns listed below.\n"
        )

    return agent.invoke({"input": question})


print_message()

user_input = st.chat_input("질문을 입력하세요.")
# warning_msg = st.empty()

if user_input:
    st.chat_message("user").write(user_input)

    response = db_research_agent(user_input)['output']
    st.chat_message("assistant").write(response)
    # with st.chat_message("assistant"):
    #     container = st.empty()

    #     # 응답 출력
    #     ai_answer = ""
    #     for token in response:
    #         ai_answer += token
    #         container.markdown(ai_answer)

    # # 대화기록 저장
    # add_message("user", user_input)
    # add_message("assistant", ai_answer)