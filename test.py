import operator
from typing import TypedDict, List, Dict, Any, Optional, Annotated, Literal
from pydantic import BaseModel, Field
from typing_extensions import Annotated
import pandas as pd
import json

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage, ChatMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import START, END
from langgraph.graph import StateGraph
from langchain.agents import create_tool_calling_agent, create_react_agent, AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain_experimental.tools import PythonREPLTool
from uuid import uuid4

from agent_config import AgentConfig
from data import DBManager
from utils import *
from tools import *
from prompts import Prompt
from states import DBState
from langgraph_builder import DB_LangGraphBuilder
from nodes import db_supervisor_node, table_selector_node, query_generator_node, data_loader_node, python_code_generator_node, python_code_executor_node
import builtins
import os
os.environ["PYDANTIC_STRICT_SCHEMA_VALIDATION"] = "False"

from dotenv import load_dotenv
from langchain_teddynote import logging

load_dotenv()
logging.langsmith("Agent-LangGraph-Test")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 시스템에 설치된 폰트 중 한글 폰트 찾기 (Mac과 Windows에서 일반적으로 사용 가능한 폰트)
try:
    # Mac OS
    font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    print(f"한글 폰트 설정: {font_prop.get_name()}")
except:
    try:
        # 나눔고딕 (많은 시스템에 설치됨)
        font_path = fm.findfont('NanumGothic')
        if font_path:
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            print(f"한글 폰트 설정: {font_prop.get_name()}")
    except:
        # 폰트를 찾지 못하면 경고만 출력
        print("경고: 한글을 지원하는 폰트를 찾을 수 없습니다. 그래프에 한글이 깨질 수 있습니다.")
        pass


# LLM 모델 설정
MODEL_NAME = "gpt-4o"  # 또는 다른 모델 이름 사용
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# 설정 및 DB 연결
conf = AgentConfig()
dbm = DBManager(conf)
db = dbm.get_db_connection()

db_builder = DB_LangGraphBuilder(DBState)
db_graph = db_builder.build_graph()

if __name__ == "__main__":
    user_query = "3월 8일 원수 탁도의 시간대별 평균을 알려줘."

    initial_state = DBState(
        messages=[{"role": "user", "content": user_query}],
        question=user_query,
        conf=conf,
        dbm=dbm,
        db=db
    )

    print("초기 상태 생성 완료")
            
    # 그래프 실행
    print("실행 시작")
    result = db_graph.invoke(initial_state)
    print("그래프 실행 완료")

    # 결과 처리 및 반환
    print("결과 반환")
    print(result)