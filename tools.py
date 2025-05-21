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

@tool
def decide_next_node(status_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    현재 상태를 분석하고 다음에 실행해야 할 노드를 결정합니다.
    
    Args:
        status_info: 시스템의 현재 상태 정보가 담긴 딕셔너리
            (has_tables, has_queries, has_data, has_python_code, has_visualization, status, question 등의 키를 포함)
            
    Returns:
        next_node와 reason 키를 포함하는 딕셔너리
    """
    if not status_info:
        # 상태 정보가 없는 경우 기본값 사용
        return {
            "next_node": "table_selector",
            "reason": "워크플로우를 시작하기 위해 테이블 선택부터 진행합니다."
        }
    
    # 상태 정보 분석
    has_tables = status_info.get("has_tables", False)
    has_queries = status_info.get("has_queries", False)
    has_data = status_info.get("has_data", False)
    has_python_code = status_info.get("has_python_code", False)
    has_visualization = status_info.get("has_visualization", False)
    status = status_info.get("status", "running")
    error_message = status_info.get("error_message", "")
    last_node = status_info.get("last_node", None)
    
    # 오류 상태 처리
    if status == "error":
        # 오류가 발생한 노드에 따라 다음 단계 결정
        if last_node == "table_selector":
            return {
                "next_node": "table_selector",
                "reason": f"테이블 선택 중 오류가 발생했습니다: {error_message}. 테이블 선택부터 다시 시도합니다."
            }
        elif last_node == "query_generator":
            return {
                "next_node": "query_generator",
                "reason": f"쿼리 생성 중 오류가 발생했습니다: {error_message}. 쿼리 생성부터 다시 시도합니다."
            }
        elif last_node == "python_code_generator":
            return {
                "next_node": "python_code_generator",
                "reason": f"Python 코드 생성 중 오류가 발생했습니다: {error_message}. 코드 생성부터 다시 시도합니다."
            }
        elif last_node == "python_code_executor":
            # Python 코드 실행 오류의 일반적인 패턴 확인
            if "result_df" in error_message:
                return {
                    "next_node": "python_code_generator",
                    "reason": "Python 코드가 result_df 변수를 생성하지 않았습니다. 코드를 다시 생성합니다."
                }
            elif "empty" in error_message.lower() or "비어" in error_message:
                return {
                    "next_node": "python_code_generator",
                    "reason": "생성된 결과 데이터프레임이 비어있습니다. 코드를 다시 생성합니다."
                }
            else:
                return {
                    "next_node": "python_code_generator",
                    "reason": f"Python 코드 실행 중 오류가 발생했습니다: {error_message}. 코드를 다시 생성합니다."
                }
        else:
            # 기본 오류 처리 - 처음부터 다시 시작
            return {
                "next_node": "table_selector",
                "reason": f"알 수 없는 오류가 발생했습니다: {error_message}. 처음부터 다시 시도합니다."
            }
    
    # 정상 상태 처리 - 워크플로우 진행
    if not has_tables:
        return {
            "next_node": "table_selector",
            "reason": "테이블이 선택되지 않았습니다. 테이블 선택부터 진행합니다."
        }
    elif not has_queries:
        return {
            "next_node": "query_generator",
            "reason": "테이블은 선택되었지만 쿼리가 없습니다. 쿼리 생성을 진행합니다."
        }
    elif has_data and not has_python_code:
        return {
            "next_node": "python_code_generator",
            "reason": "데이터는 로드되었지만 분석 코드가 없습니다. Python 코드 생성을 진행합니다."
        }
    elif has_data and has_python_code:
        return {
            "next_node": None,
            "reason": "모든 단계가 완료되었습니다. 워크플로우를 종료합니다."
        }
    
    # 기본값 (처리할 수 없는 상태)
    return {
        "next_node": None,
        "reason": "현재 상태에서 수행할 다음 작업을 결정할 수 없습니다."
    }

@tool
def generate_final_answer(status_info: Dict[str, Any]) -> str:
    """
    분석 결과를 기반으로 사용자에게 제공할 최종 답변을 생성합니다.
    
    Args:
        status_info: 시스템의 현재 상태 정보가 담긴 딕셔너리
            (result_df, has_visualization, visualization_data, status, question, python_code 등의 키를 포함)
            
    Returns:
        사용자에게 표시할 최종 답변 메시지
    """
    # 상태 확인
    if not status_info:
        return "분석 상태 정보가 없습니다."
        
    status = status_info.get("status", "")
    question = status_info.get("question", "")
    
    # completed 상태 확인 - 최종 답변 생성
    if status == "completed":
        # 이미 생성된 final_answer가 있는지 확인
        if "final_answer" in status_info and status_info["final_answer"]:
            return status_info["final_answer"]
            
        # 코드와 결과를 분석하여 포괄적인 답변 생성
        python_code = status_info.get("python_code", "")
        
        # 분석 결과 해석
        analysis_summary = ""
        result_df = None
        if "result_df" in status_info:
            try:
                result_df = status_info["result_df"]
                if hasattr(result_df, "shape"):
                    rows, cols = result_df.shape
                    analysis_summary += f"\n분석 결과는 {rows}행 {cols}열의 데이터를 포함합니다. "
                    
                    # 결과 데이터프레임 내용 확인
                    if rows > 0:
                        if cols <= 5:  # 열이 적은 경우 모든 열 포함
                            analysis_summary += f"\n주요 열: {', '.join(str(col) for col in result_df.columns)}"
                        else:  # 열이 많은 경우 주요 열만 표시
                            analysis_summary += f"\n주요 열 일부: {', '.join(str(col) for col in result_df.columns[:5])}..."
            except:
                pass
        
        # 시각화 결과 확인 - 직접 상태에서 확인
        has_visualization = status_info.get("has_visualization", False)
        visualization_data = status_info.get("visualization_data", None)
        
        if has_visualization:
            analysis_summary += "\n시각화 결과가 포함되어 있습니다."
        
        # LLM을 사용하여 자연스러운 답변 생성
        try:
            # 프롬프트 및 LLM 생성
            from prompts import Prompt
            from langchain_openai.chat_models import ChatOpenAI
            from langchain_core.output_parsers import StrOutputParser
            
            llm_for_answer = ChatOpenAI(model=MODEL_NAME, temperature=0.3)
            answer_prompt = Prompt().final_answer_prompt()
            
            # 결과 데이터의 요약 생성
            data_summary = ""
            if result_df is not None and not result_df.empty:
                try:
                    # 데이터 요약을 위한 기본 정보 추출
                    data_summary += f"데이터 형태: {result_df.shape}\n"
                    data_summary += f"컬럼: {list(result_df.columns)}\n"
                    
                    # 첫 5개 행 정보 추가
                    if len(result_df) > 0:
                        data_summary += "데이터 샘플:\n"
                        sample_data = result_df.head(5).to_string()
                        data_summary += sample_data
                except Exception as e:
                    data_summary += f"데이터 요약 중 오류: {str(e)}"
            
            # 시각화 데이터 정보 추가
            visualization_info = ""
            if visualization_data:
                visualization_info = "시각화 데이터가 포함되어 있습니다."
            
            # LLM 체인 실행
            final_answer_chain = answer_prompt | llm_for_answer | StrOutputParser()
            final_answer = final_answer_chain.invoke({
                "question": question,
                "python_code": python_code,
                "analysis_summary": analysis_summary,
                "data_summary": data_summary,
                "has_visualization": has_visualization,
                "visualization_info": visualization_info
            })
            
            return final_answer
            
        except Exception as e:
            # LLM 체인에 오류 발생 시 기본 포맷 사용
            print(f"LLM 답변 생성 중 오류: {str(e)}")
            
            # 기본 포맷으로 답변 생성
            answer = f"질문: {question}\n\n"
            answer += f"답변: "
            
            # 기본 분석 정보 추가
            if result_df is not None:
                try:
                    if not result_df.empty:
                        answer += f"\n데이터 분석 결과:\n"
                        sample_data = result_df.head(5).to_string()
                        answer += sample_data
                except Exception as e:
                    answer += f"\n데이터 처리 중 오류: {str(e)}"
            
            if analysis_summary:
                answer += f"\n\n추가 정보: {analysis_summary}"
            
            return answer
    
    # completed 상태가 아닌 경우 - 오류 또는 진행 중
    if status == "error":
        error_message = status_info.get("error_message", "")
        return f"죄송합니다. 분석 중 오류가 발생했습니다: {error_message}"
    
    # 진행 중인 경우
    return "아직 분석이 완료되지 않았습니다. 처리 중입니다..."

# @tool
# def analyze_error(status_info: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     오류를 분석하고 해결 방안을 제시합니다.
    
#     Args:
#         status_info: 시스템의 현재 상태 정보가 담긴 딕셔너리
#             (error, status, last_node 등의 키를 포함)
            
#     Returns:
#         오류 분석 결과와 해결 방안이 담긴 딕셔너리
#     """
#     # 오류 정보 확인
#     status = status_info.get("status", "")
#     last_node = status_info.get("last_node", "")
#     error_message = status_info.get("error_message", "")
    
#     if not error_message and "error" in status_info:
#         error_message = status_info["error"].get("message", "")
    
#     # 오류가 없는 경우
#     if status != "error" or not error_message:
#         return {
#             "error_node": None,
#             "error_analysis": "오류가 발생하지 않았습니다.",
#             "solution": "정상적으로 처리가 진행 중입니다.",
#             "retry_node": None
#         }
    
#     # 노드별 오류 분석 및 해결 방안
#     error_node = last_node if last_node else "unknown"
    
#     # 데이터베이스 연결 오류
#     if "데이터베이스 연결" in error_message:
#         return {
#             "error_node": error_node,
#             "error_analysis": "데이터베이스 연결에 실패했습니다.",
#             "solution": "데이터베이스 연결 상태를 확인하고 재시도하세요.",
#             "retry_node": "table_selector"  # 처음부터 다시 시작
#         }
    
#     # 테이블 선택 오류
#     elif error_node == "table_selector":
#         return {
#             "error_node": "table_selector",
#             "error_analysis": f"테이블 선택 중 오류가 발생했습니다: {error_message}",
#             "solution": "사용자 질문과 관련된 테이블을 다시 선택해보세요.",
#             "retry_node": "table_selector"
#         }
    
#     # 쿼리 생성 오류
#     elif error_node == "query_generator":
#         return {
#             "error_node": "query_generator",
#             "error_analysis": f"SQL 쿼리 생성 중 오류가 발생했습니다: {error_message}",
#             "solution": "테이블 스키마를 확인하고 SQL 쿼리를 다시 생성해보세요.",
#             "retry_node": "query_generator"
#         }
    
#     # 데이터 로드 오류
#     elif error_node == "data_loader":
#         return {
#             "error_node": "data_loader",
#             "error_analysis": f"데이터 로드 중 오류가 발생했습니다: {error_message}",
#             "solution": "SQL 쿼리 실행에 문제가 있습니다. 쿼리를 다시 생성해보세요.",
#             "retry_node": "query_generator"  # 쿼리 생성부터 다시 시작
#         }
    
#     # Python 코드 생성 오류
#     elif error_node == "python_code_generator":
#         return {
#             "error_node": "python_code_generator",
#             "error_analysis": f"Python 코드 생성 중 오류가 발생했습니다: {error_message}",
#             "solution": "데이터 분석을 위한 Python 코드를 다시 생성해보세요.",
#             "retry_node": "python_code_generator"
#         }
    
#     # Python 코드 실행 오류
#     elif error_node == "python_code_executor":
#         # Python 코드 실행 오류의 일반적인 패턴 확인
#         if "result_df" in error_message:
#             return {
#                 "error_node": "python_code_executor",
#                 "error_analysis": "Python 코드가 result_df 변수를 생성하지 않았습니다.",
#                 "solution": "코드 마지막에 반드시 'result_df = ...' 형태로 결과를 저장하도록 코드를 수정하세요.",
#                 "retry_node": "python_code_generator"
#             }
#         elif "empty" in error_message.lower() or "비어" in error_message:
#             return {
#                 "error_node": "python_code_executor",
#                 "error_analysis": "생성된 결과 데이터프레임이 비어있습니다.",
#                 "solution": "데이터 필터링 조건을 확인하고, 결과가 비어있지 않도록 코드를 수정하세요.",
#                 "retry_node": "python_code_generator"
#             }
#         else:
#             return {
#                 "error_node": "python_code_executor",
#                 "error_analysis": f"Python 코드 실행 중 오류가 발생했습니다: {error_message}",
#                 "solution": "코드의 오류를 수정하고 다시 실행해보세요.",
#                 "retry_node": "python_code_generator"
#             }
    
#     # 기본 오류 처리
#     else:
#         return {
#             "error_node": error_node,
#             "error_analysis": f"알 수 없는 오류가 발생했습니다: {error_message}",
#             "solution": "처음부터 다시 시도해보세요.",
#             "retry_node": "table_selector"  # 처음부터 다시 시작
#         }