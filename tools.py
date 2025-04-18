from typing import TypedDict, List, Dict, Any, Optional, Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolNode
from langchain_openai.chat_models import ChatOpenAI

from agent_config import AgentConfig
from prompts import Prompt
from utils import json_parse, process_ai_rt, convert_value, process_schema_for_prompt

import sys
import io
import pandas as pd


MODEL_NAME = "gpt-4o"

class CustomPythonREPLTool(BaseTool):
    name: str = "python_repl_tool"
    description: str = "A Python REPL. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."
    
    def _run(self, command: str) -> str:
        """Run command with own globals/locals and returns the output."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        
        try:
            exec(command, globals())
            sys.stdout = old_stdout
            output = mystdout.getvalue()
            return output
        except Exception as e:
            sys.stdout = old_stdout
            return f"Error: {str(e)}"

# 오류 처리 함수
def handle_tool_error(state) -> dict:
    # 오류 정보 조회
    error = state.get("error")
    # 도구 정보 조회
    tool_calls = state["messages"][-1].tool_calls
    # ToolMessage 로 래핑 후 반환
    return {
        "messages": [
            ToolMessage(
                content=f"Here is the error: {repr(error)}\n\nPlease fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


# 오류를 처리하고 에이전트에 오류를 전달하기 위한 ToolNode 생성
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    # 오류 발생 시 대체 동작을 정의하여 ToolNode에 추가
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

@tool
def select_table(question: str) -> Dict[str, str]:
    """
    사용자 쿼리를 기반으로 데이터베이스에서 관련 테이블을 선택합니다.
    """
    conf = AgentConfig()
    AVAILABLE_TABLES = f"""
        "TB_{conf.algorithm_code}_RT": "실시간 측정 데이터 테이블. 센서의 실시간 측정값을 저장합니다.",
        "TB_AI_{conf.algorithm_code}_RT": "AI 분석 결과 테이블. AI가 분석한 예측값과 결과를 저장합니다.",
        "TB_AI_{conf.algorithm_code}_CTR": "AI 제어 결과 테이블. AI의 제어 명령과 결과를 저장합니다.",
        "TB_AI_{conf.algorithm_code}_ALM": "AI 경고 테이블. AI운영 도중 발생한 알람 메시지를 저장합니다.",
        "TB_AI_{conf.algorithm_code}_INIT": "AI 알고리즘의 초기 설정 테이블. 상한, 하한 및 초기 설정 값을 저장",
        "TB_TAG_MNG": "태그 관리 테이블. 시스템에서 사용하는 태그(센서 등)의 메타데이터를 저장합니다."
    """

    # 프롬프트 생성
    table_selection_prompt = Prompt().table_selection_prompt()

    llm_for_table_selection = ChatOpenAI(model=MODEL_NAME, temperature=0)

    # LLM 체인 실행
    select_table_chain = table_selection_prompt | llm_for_table_selection | StrOutputParser()
    response = select_table_chain.invoke({
                                        "question": question,
                                        "available_tables": AVAILABLE_TABLES
                                    })
    
    return json_parse(response)

@tool
def generate_sql_query(
        question: str,
        selected_tables: Dict[str, str],
        db: Any = None
) -> Dict[str, str]:
    """    
    사용자의 자연어 쿼리를 분석하여 지정된 테이블에 대한 SQL 문을 생성하고, 
    테이블 이름과 해당 테이블에 대한 SQL 문을 반환합니다.
    """
    # db 매개변수가 없으면 builtins에서 가져옴
    import builtins
    db = getattr(builtins, 'db', None)
        
    # db가 여전히 None이면 오류 발생
    if db is None:
        raise ValueError("Database connection not available")
        
    TABLE_SCHEMAS = {
            table: db.get_table_info([table]) for table in selected_tables.keys()
        }
    schema_info = ""
    for table, schema in TABLE_SCHEMAS.items():
        schema_info += f"\n## {table} Table:\n{process_schema_for_prompt(schema)}\n"
    
    # SQL 쿼리 생성 프롬프트
    sql_query_generation_prompt = Prompt().sql_query_generation_prompt()

    llm_for_sql_query = ChatOpenAI(model=MODEL_NAME, temperature=0)
    
    # LLM 체인 실행
    sql_query_generator_chain = sql_query_generation_prompt | llm_for_sql_query | StrOutputParser()
    query_dict = {}
    for table_name in selected_tables.keys():
        sql_query = sql_query_generator_chain.invoke({
                    "schema_info": schema_info,
                    "question": question,
                    "table_name": table_name
                })
        query_dict[table_name] = sql_query

    return query_dict

@tool
def save_figure(file_path: str) -> str:
    """
    현재 matplotlib 그림을 지정된 경로에 저장합니다.
    
    Args:
        file_path: 저장할 파일 경로 (예: 'output.png')
        
    Returns:
        저장된 파일 경로
    """
    import matplotlib.pyplot as plt
    import os
    
    # 이미지 저장 디렉토리 생성
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    # 현재 그림 저장
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    return f"이미지가 {file_path}에 저장되었습니다."

# @tool
def load_data(
        query_dict: Annotated[dict, generate_sql_query],
        dbm: Any
) -> Dict[str, pd.DataFrame]:
    """
    테이블 이름을 입력하면 테이블의 데이터를 불러옵니다.
    """

    df_dict = {}
    for table, query in query_dict.items():
        df = dbm.select_from_table(query)
        df = convert_df(table, df)
        df_dict[table] = df

    return df_dict

def convert_df(table_name: str, df: pd.DataFrame):
    """
    테이블 이름에 따라 데이터 프레임을 변환합니다.
    """
    if table_name == 'TB_AI_{conf.algorithm_code}_RT':
        df = process_ai_rt(df)
    else:
        df = df.map(convert_value)

    return df