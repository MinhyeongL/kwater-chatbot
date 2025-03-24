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
from datetime import datetime
from data import DBManager
from agent_config import AgentConfig

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks

from typing import List, Union


# API KEY 정보로드
load_dotenv()

# LangSmith 로깅
logging.langsmith("KWATER 챗봇")

# 페이지 설정
st.set_page_config(page_title="KWATER 챗봇", page_icon="💧")

# 페이지 제목
st.title("KWATER 챗봇(데모용)")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # 대화 내용을 저장할 리스트 초기화

if "conf" not in st.session_state:
    conf = AgentConfig(
        location_code="A",
        plant_code="SN",
        algorithm_code="C"
    )
    st.session_state["conf"] = conf

if "dbm" not in st.session_state:
    st.session_state["dbm"] = DBManager(st.session_state["conf"])

if "db" not in st.session_state:
    st.session_state["db"] = st.session_state["dbm"].get_db_connection()


# 상수 정의
class MessageRole:
    """
    메시지 역할을 정의하는 클래스입니다.
    """

    USER = "user"  # 사용자 메시지 역할
    ASSISTANT = "assistant"  # 어시스턴트 메시지 역할


class MessageType:
    """
    메시지 유형을 정의하는 클래스입니다.
    """

    TEXT = "text"  # 텍스트 메시지
    FIGURE = "figure"  # 그림 메시지
    CODE = "code"  # 코드 메시지
    DATAFRAME = "dataframe"  # 데이터프레임 메시지


# 메시지 관련 함수
def print_messages():
    """
    저장된 메시지를 화면에 출력하는 함수입니다.
    """
    for role, content_list in st.session_state["messages"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):
                    message_type, message_content = content
                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)  # 텍스트 메시지 출력
                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content)  # 그림 메시지 출력
                    elif message_type == MessageType.CODE:
                        with st.status("코드 출력", expanded=False):
                            st.code(
                                message_content, language="python"
                            )  # 코드 메시지 출력
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content)  # 데이터프레임 메시지 출력
                else:
                    raise ValueError(f"알 수 없는 콘텐츠 유형: {content}")


def add_message(role: MessageRole, content: List[Union[MessageType, str]]):
    """
    새로운 메시지를 저장하는 함수입니다.

    Args:
        role (MessageRole): 메시지 역할 (사용자 또는 어시스턴트)
        content (List[Union[MessageType, str]]): 메시지 내용
    """
    messages = st.session_state["messages"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])  # 같은 역할의 연속된 메시지는 하나로 합칩니다
    else:
        messages.append([role, [content]])  # 새로운 역할의 메시지는 새로 추가합니다


with st.sidebar:
    clear_btn = st.button("대화 초기화")  # 대화 내용을 초기화하는 버튼

    selected_model = st.selectbox(
        "OpenAI 모델을 선택해주세요.", ["gpt-4o", "gpt-4o-mini"], index=0
    )


# 콜백 함수
def tool_callback(tool):
    """
    도구 실행 결과를 처리하는 콜백 함수입니다.

    Args:
        tool (dict): 실행된 도구 정보
    """
    result = ""

    if tool_name := tool.get("tool"):
        if tool_name == "python_repl_ast":
            tool_input = tool.get("tool_input", {})
            query = tool_input.get("query")
            if query:

                # # 멀티인덱스 데이터프레임을 위한 쿼리 전처리
                # query = preprocess_query(query)
                # st.write(query)

                df_in_result = None
                with st.status("데이터 분석 중...", expanded=True) as status:
                    st.markdown(f"```python\n{query}\n```")
                    add_message(MessageRole.ASSISTANT, [MessageType.CODE, query])
                    if "df" in st.session_state:
                        result = st.session_state["python_tool"].invoke(
                            {"query": query}
                        )
                        if isinstance(result, pd.DataFrame):
                            df_in_result = result
                    status.update(label="코드 출력", state="complete", expanded=False)

                if df_in_result is not None:
                    # st.dataframe(df_in_result)
                    add_message(
                        MessageRole.ASSISTANT, [MessageType.DATAFRAME, df_in_result]
                    )

                if "plt.show" in query:
                    plt.rc('font', family='AppleGothic')
                    plt.rc('axes', unicode_minus=False)

                    # 먼저 쿼리 실행하여 그래프 생성
                    result = st.session_state["python_tool"].invoke({"query": query})
                    # st.write(st.session_state["df"].head())

                    fig = plt.gcf()
                    ax = plt.gca()

                    # # 데이터 존재 여부 확인
                    # if len(ax.get_lines()) == 0:
                    #     st.error("그래프에 데이터가 없습니다. 쿼리를 확인해주세요.")
                    #     return

                    # 선 색상 설정 (기존 선 스타일 유지하면서 색상만 변경)
                    for line in ax.get_lines():
                        line.set_color('orange')

                    # 축과 눈금 설정
                    for spine in ax.spines.values():
                        spine.set_color('black')

                    # 눈금 레이블 색상
                    ax.tick_params(axis='both', colors='black')

                    # 축 레이블과 제목 색상
                    ax.xaxis.label.set_color('black')
                    ax.yaxis.label.set_color('black')
                    if ax.get_title():
                        ax.title.set_color('black')

                    # # 축과 눈금 설정
                    # ax.spines['bottom'].set_color('white')
                    # ax.spines['top'].set_color('white')
                    # ax.spines['left'].set_color('white')
                    # ax.spines['right'].set_color('white')

                    # 그래프 사이즈 조정
                    fig.set_size_inches(12, 6)
                    plt.tight_layout()

                    st.pyplot(fig, clear_figure=True)
                    add_message(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])

                    plt.show()
                    # st.pyplot(plt.gcf())
                    # st.pyplot(fig)
                    plt.close("all")
                return result
            else:
                st.error(
                    "데이터프레임이 정의되지 않았습니다. CSV 파일을 먼저 업로드해주세요."
                )
                return result
            

def observation_callback(observation) -> None:
    """
    관찰 결과를 처리하는 콜백 함수입니다.

    Args:
        observation (dict): 관찰 결과
    """
    if "observation" in observation:
        obs = observation["observation"]
        if isinstance(obs, str) and "Error" in obs:
            st.error(obs)
            st.session_state["messages"][-1][
                1
            ].clear()  # 에러 발생 시 마지막 메시지 삭제


def result_callback(result: str) -> None:
    """
    최종 결과를 처리하는 콜백 함수입니다.

    Args:
        result (str): 최종 결과
    """
    pass  # 현재는 아무 동작도 하지 않습니다


def extract_tables(llm_response: str) -> List[str]:
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
    if "TB_TAG_MNG" not in valid_tables:
        valid_tables.append("TB_TAG_MNG")
    return valid_tables


def convert_value(value):
    # None 값은 그대로 반환
    if value is None:
        return value

    # 이미 Timestamp 타입인 경우 처리
    if isinstance(value, pd.Timestamp):
        return value

    # 이미 숫자 타입인 경우 처리
    if isinstance(value, (int, float)):
        return value

    # 문자열만 변환 시도
    if isinstance(value, str):
        # 숫자 변환 시도
        try:
            num_value = float(value)
            return num_value
        except ValueError:
            pass

        # 날짜 변환 시도
        try:
            date_value = datetime.strptime(value, '%Y-%m-%d')
            return date_value
        except ValueError:
            pass

    # 변환 실패 시 원래 값 반환
    return value


# 에이전트 생성 함수
def create_agent(dataframe, selected_model="gpt-4o"):
    """
    데이터프레임 에이전트를 생성하는 함수입니다.

    Args:
        dataframe (pd.DataFrame): 분석할 데이터프레임
        selected_model (str, optional): 사용할 OpenAI 모델. 기본값은 "gpt-4o"

    Returns:
        Agent: 생성된 데이터프레임 에이전트
    """
    llm = ChatOpenAI(model=selected_model, temperature=0)

    # Python REPL 도구 생성 시 globals와 locals를 명시적으로 설정
    python_repl = PythonAstREPLTool(
        globals={"pd": pd, "plt": plt, "np": np, "sns": sns, "df": dataframe.copy()},  # 필요한 모듈들을 globals에 추가
        locals=None
        # locals={"df": dataframe}  # 데이터프레임을 locals에 추가
    )
    if "python_tool" not in st.session_state:
        st.session_state["python_tool"] = python_repl

    return create_pandas_dataframe_agent(
            llm,
            dataframe,
            verbose=False,
            agent_type="tool-calling",
            allow_dangerous_code=True,
            extra_tools=[python_repl],
            prefix="You are a professional data analyst and expert in Pandas. "
            "You must use Pandas DataFrame(`df`) to answer user's request. "
            "\n\n[IMPORTANT] DO NOT create or overwrite the `df` variable in your code. \n\n"
            "You don't need to generate visualization code."
            "When creating visualizations, ALWAYS include `plt.show()` at the end of your code. "
            "I prefer seaborn code for visualization, but you can use matplotlib as well."
            "\n\n<Visualization Preference>\n"
            "- [IMPORTANT] Use `Korean` for your visualization title and labels."
            "The language of final answer should be written in Korean. "
            "\n\n###\n\n<Column Guidelines>\n"
            "If user asks with columns that are not listed in `df.columns`, you may refer to the most similar columns listed below.\n"
        )


# 질문 처리 함수
def ask(query):
    """
    사용자의 질문을 처리하고 응답을 생성하는 함수입니다.

    Args:
        query (str): 사용자의 질문
    """
    # if "agent" in st.session_state:
    llm = ChatOpenAI(model=selected_model, temperature=0)

    st.chat_message("user").write(query)
    add_message(MessageRole.USER, [MessageType.TEXT, query])

    prompt = Prompt()
    table_selection_prompt = prompt.table_selection_prompt()
    select_table_chain = table_selection_prompt | llm | StrOutputParser()
    response = select_table_chain.invoke({"question": query})
    selected_tables = extract_tables(response)

    TABLE_SCHEMAS = {
        table: st.session_state["db"].get_table_info([table]) for table in selected_tables
    }

    sql_query_gen_prompt = prompt.sql_query_gen_prompt(TABLE_SCHEMAS)
    sql_query_gen_chain = sql_query_gen_prompt | llm | StrOutputParser()

    sql_query = sql_query_gen_chain.invoke({"question": query})

    dataframe = st.session_state["dbm"].select_from_table(sql_query)
    dataframe = dataframe.applymap(convert_value)

    agent = create_agent(dataframe)
    response = agent.stream({"input": query})

    ai_answer = ""
    parser_callback = AgentCallbacks(
        tool_callback, observation_callback
    )
    stream_parser = AgentStreamParser(parser_callback)

    assistant_container = st.empty()
    with assistant_container.container():
        with st.chat_message("assistant"):
            for step in response:
                stream_parser.process_agent_steps(step)
                if "output" in step:
                    ai_answer += step["output"]
            st.write(ai_answer)

    add_message(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])


# 메인 로직
if clear_btn:
    st.session_state["messages"] = []  # 대화 내용 초기화


user_input = st.chat_input("질문을 입력하세요.")

if user_input:
    ask(user_input)  # 사용자 질문 처리
