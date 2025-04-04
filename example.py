from typing import List, Union
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import *

import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

st.set_page_config(layout="wide")
# API 키 및 프로젝트 설정
load_dotenv()
# logging.langsmith("CSV Agent 챗봇")

# Streamlit 앱 설정
st.title("Welcome to Laboratory Page")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # 대화 내용을 저장할 리스트 초기화


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


# 사이드바 설정
with st.sidebar:
    clear_btn = st.button("대화 초기화")  # 대화 내용을 초기화하는 버튼
    # uploaded_file = st.file_uploader(
    #     "CSV 파일을 업로드 해주세요.", type=["csv"], accept_multiple_files=True
    # )  # CSV 파일 업로드 기능
    selected_model = st.selectbox(
        "OpenAI 모델을 선택해주세요.", ["gpt-4o", "gpt-4o-mini"], index=1
    )  # OpenAI 모델 선택 옵션


    if "start_date_lab" not in st.session_state:
        st.session_state["start_date_lab"] = st.session_state["ai_rt"].index[-1].date() - timedelta(days=1)
    if "end_date_lab" not in st.session_state:
        st.session_state["end_date_lab"] = st.session_state["ai_rt"].index[-1].date()

    start = st.date_input('시작 날짜', st.session_state["start_date_lab"])
    end = st.date_input('종료 날짜', st.session_state["end_date_lab"])
    lab_df = load_ai_rt_op(start, end)



    apply_btn = st.button("데이터 분석 시작")  # 데이터 분석을 시작하는 버튼


def examlple_figure(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set the style for the plot
    sns.set_theme(style="white", palette="muted")

    # Plot the trend of 'B_TE'
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y=('B_TE', 0, 0, 0, 0), color='b')
    plt.title('Trend of B_TE')
    plt.xlabel('Time')
    plt.ylabel('B_TE Value')
    plt.show()

# st.pyplot(examlple_figure(lab_df))

# 콜백 함수
def tool_callback(tool) -> None:
    """
    도구 실행 결과를 처리하는 콜백 함수입니다.

    Args:
        tool (dict): 실행된 도구 정보
    """
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
                    # 먼저 쿼리 실행하여 그래프 생성
                    result = st.session_state["python_tool"].invoke({"query": query})
                    # st.write(st.session_state["df"].head())

                    fig = plt.gcf()
                    ax = plt.gca()

                    # # 데이터 존재 여부 확인
                    # if len(ax.get_lines()) == 0:
                    #     st.error("그래프에 데이터가 없습니다. 쿼리를 확인해주세요.")
                    #     return
                    
                    fig.patch.set_facecolor('#212750')  # 그래프 영역 배경색
                    ax.set_facecolor('#212750')  # 플롯 영역 배경색

                    # 선 색상 설정 (기존 선 스타일 유지하면서 색상만 변경)
                    for line in ax.get_lines():
                        line.set_color('orange')

                    # 축과 눈금 설정
                    for spine in ax.spines.values():
                        spine.set_color('white')
                        
                    # 눈금 레이블 색상
                    ax.tick_params(axis='both', colors='white')

                    # 축 레이블과 제목 색상
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    if ax.get_title():
                        ax.title.set_color('white')
                        
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
                    plt.close("all")
                return result
            else:
                st.error(
                    "데이터프레임이 정의되지 않았습니다. CSV 파일을 먼저 업로드해주세요."
                )
                return


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
        "If you are willing to generate visualization code, please use `plt.show()` at the end of your code. "
        "I prefer seaborn code for visualization, but you can use matplotlib as well."
        "\n\n<Visualization Preference>\n"
        "- [IMPORTANT] Use `English` for your visualization title and labels."
        "- `muted` cmap, white background, and no grid for your visualization."
        "\nRecommend to set cmap, palette parameter for seaborn plot if it is applicable. "
        "The language of final answer should be written in Korean. "
        "\n\n###\n\n<Column Guidelines>\n"
        "If user asks with columns that are not listed in `df.columns`, you may refer to the most similar columns listed below.\n"
        "If the columns are in a multi-index format, please consider the form (ITM, STG, SER, LOC, STP). "
        "STG represents '단계', SER represents '계열', LOC represents '지', and STP represents '스텝'. "
        "For example, 3단계 E_TB_B, it would be ('E_TB_B', 3, 0, 0, 0).\n"
    )


# 질문 처리 함수
def ask(query):
    """
    사용자의 질문을 처리하고 응답을 생성하는 함수입니다.

    Args:
        query (str): 사용자의 질문
    """
    if "agent" in st.session_state:
        st.chat_message("user").write(query)
        add_message(MessageRole.USER, [MessageType.TEXT, query])

        agent = st.session_state["agent"]
        response = agent.stream({"input": query})

        ai_answer = ""
        parser_callback = AgentCallbacks(
            tool_callback, observation_callback, result_callback
        )
        stream_parser = AgentStreamParser(parser_callback)

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



# st.dataframe(lab_df.tail())

def examlple_figure(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # B_TE 데이터 추출
    b_te_data = df[('B_TB', 0, 0, 0, 0)]

    # 시계열 데이터로 인덱스 설정
    b_te_data.index = df.index

    # 시각화
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=b_te_data.index, y=b_te_data.values, palette='muted')
    plt.title('Trend of B_TE over Time')
    plt.xlabel('Time')
    plt.ylabel('B_TE Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# st.dataframe(lab_df.head(1))
# st.dataframe(lab_df.tail(1))
# st.pyplot(examlple_figure(lab_df))

if apply_btn:

    st.session_state["df"] = lab_df  # 데이터프레임 저장   
    st.session_state["agent"] = create_agent(
        lab_df, selected_model
    )  # 에이전트 생성
    st.success("설정이 완료되었습니다. 대화를 시작해 주세요!")
elif apply_btn:
    st.warning("파일을 업로드 해주세요.")

print_messages()  # 저장된 메시지 출력

user_input = st.chat_input("궁금한 내용을 물어보세요!")  # 사용자 입력 받기
if user_input:
    ask(user_input)  # 사용자 질문 처리
