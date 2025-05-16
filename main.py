"""
개선된 LangGraph 그래프 구현 모듈입니다.
직렬화 오류 처리와 에이전트 연동을 개선하였습니다.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, Annotated
import pandas as pd
import json
import builtins
import traceback
from uuid import uuid4
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool, tool
from langgraph.graph import StateGraph, END

# 프로젝트 내부 임포트
from agent_config import AgentConfig
from data import DBManager
from tools import generate_sql_query, select_table, CustomPythonREPLTool
from serializer_utils import sanitize_state_for_serialization, sanitize_intermediate_steps
from agents import run_reaction_agent, update_state_with_reaction_results
from utils import json_parse, format_tables_for_prompt
from prompts import Prompt


MODEL_NAME = "gpt-4o"


# 상태 모델 정의 (Pydantic v2 방식 사용)
class DBState(BaseModel):
    # 대화 히스토리
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    question: Optional[str] = None
    
    # 각 도구의 출력
    selected_tables: Dict[str, str] = Field(default_factory=dict)
    generated_queries: Dict[str, str] = Field(default_factory=dict)
    df_dict: Dict[str, Any] = Field(default_factory=dict, exclude=True)  # 직렬화에서 제외
    python_code: Optional[str] = None  # 생성된 파이썬 코드 저장
    final_result: Optional[str] = None
    result_df: Optional[Any] = Field(default=None, exclude=True)  # 직렬화에서 제외
    
    # 상태 관리
    status: str = "running"
    error: Optional[Dict[str, Any]] = None
    
    # 중간 단계 기록
    intermediate_steps: List[Dict[str, Any]] = Field(default_factory=list)

    # config
    conf: Optional[Any] = Field(default=None, exclude=True)  # AgentConfig
    dbm: Optional[Any] = Field(default=None, exclude=True)  # DBManager
    db: Optional[Any] = Field(default=None, exclude=True)  # DB Connection
    
    # Pydantic v2 설정
    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {
            pd.DataFrame: lambda _: "<DataFrame>"
        }
    }
    
    def model_dump(self, **kwargs):
        """
        DataFrame 등을 제외하고 직렬화하는 메서드
        개선된 방식으로 직렬화 처리
        """
        # 무한 재귀 방지: 먼저 기본 model_dump 호출
        data = super().model_dump(**kwargs)
        
        # DataFrame 관련 필드 제거
        for field in ['df_dict', 'result_df', 'db', 'dbm', 'conf']:
            if field in data:
                data.pop(field)
        
        # 중간 단계 처리 (깊은 직렬화)
        if 'intermediate_steps' in data and data['intermediate_steps']:
            data['intermediate_steps'] = sanitize_intermediate_steps(data['intermediate_steps'])
        
        return data


# 노드 함수들
def table_selector_node(state: DBState) -> Dict[str, Any]:
    """
    테이블 선택 노드: 사용자 질문을 기반으로 관련 테이블을 선택합니다.
    """
    question = state.question if state.question else ""
    
    try:
        # 도구 정의
        tools = [select_table]
        
        # 프롬프트 생성
        table_selector_prompt = Prompt().table_selector_prompt()
        
        # 에이전트 생성
        agent = create_tool_calling_agent(
            llm=ChatOpenAI(model=MODEL_NAME, temperature=0), 
            tools=tools, 
            prompt=table_selector_prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            verbose=True
        )
        
        # 메시지 형식 변환
        messages_for_agent = []
        for msg in state.messages:
            if msg.get("role") in ["user", "human"]:
                messages_for_agent.append({"role": "user", "content": msg.get("content", "")})
            elif msg.get("role") in ["assistant", "ai"]:
                messages_for_agent.append({"role": "assistant", "content": msg.get("content", "")})
            elif msg.get("role") == "system":
                messages_for_agent.append({"role": "system", "content": msg.get("content", "")})
        
        # 에이전트 실행
        result = agent_executor.invoke({
            "question": question,
            "messages": messages_for_agent,
            "tools": ", ".join([t.name for t in tools]),
            "tool_names": ", ".join([t.name for t in tools])
        })
        
        # 결과 처리
        selected_tables = json_parse(result["output"])
        
        # 상태 업데이트
        new_state = sanitize_state_for_serialization(state)
        new_state["selected_tables"] = selected_tables
        
        # 메시지 추가
        assistant_message = {"role": "assistant", "content": "테이블 선택을 완료했습니다."}
        tool_message = {
            "role": "tool", 
            "content": result["output"],
            "name": "select_table",
            "tool_call_id": str(uuid4())
        }
        new_state["messages"] = state.messages + [assistant_message, tool_message]
        
        return new_state
        
    except Exception as e:
        # 오류 처리
        print(f"Error in table_selector_node: {str(e)}")
        traceback.print_exc()
        
        return {
            **sanitize_state_for_serialization(state),
            "error": {"message": str(e)},
            "status": "error"
        }


def query_generator_node(state: DBState) -> Dict[str, Any]:
    """
    쿼리 생성 노드: 테이블 선택 결과를 바탕으로 SQL 쿼리를 생성합니다.
    """
    # 상태 확인
    if not state.selected_tables:
        return {**sanitize_state_for_serialization(state), "error": {"message": "선택된 테이블이 없습니다."}, "status": "error"}
    
    question = state.question if state.question else ""
    selected_tables = state.selected_tables
    db = state.db
    
    try:
        # 전역 네임스페이스에 DB 설정
        builtins.db = db
        
        # 도구 정의
        tools = [generate_sql_query]
        
        # 프롬프트 생성
        query_generator_prompt = Prompt().query_generator_prompt()
        
        # 에이전트 생성
        agent = create_tool_calling_agent(
            llm=ChatOpenAI(model=MODEL_NAME, temperature=0), 
            tools=tools, 
            prompt=query_generator_prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            verbose=True
        )
        
        # 선택된 테이블 형식화
        tables_formatted = format_tables_for_prompt(selected_tables)
        selected_tables_json = json.dumps(selected_tables, indent=2)
        
        # 메시지 형식 변환
        messages_for_agent = []
        for msg in state.messages:
            if msg.get("role") in ["user", "human"]:
                messages_for_agent.append({"role": "user", "content": msg.get("content", "")})
            elif msg.get("role") in ["assistant", "ai"]:
                messages_for_agent.append({"role": "assistant", "content": msg.get("content", "")})
            elif msg.get("role") == "system":
                messages_for_agent.append({"role": "system", "content": msg.get("content", "")})
        
        # 에이전트 실행
        result = agent_executor.invoke({
            "question": question,
            "messages": messages_for_agent,
            "selected_tables": selected_tables,  # 중요! 이 매개변수 필수
            "tables_formatted": tables_formatted,  # 프롬프트에 표시할 형식화된 문자열
            "selected_tables_json": selected_tables_json,  # 프롬프트에 표시할 JSON 예시
            "tools": ", ".join([t.name for t in tools]),
            "tool_names": ", ".join([t.name for t in tools])
        })
        
        # 결과 처리
        generated_queries = json_parse(result["output"])
        
        # 상태 업데이트
        new_state = sanitize_state_for_serialization(state)
        new_state["generated_queries"] = generated_queries
        
        # 메시지 추가
        assistant_message = {"role": "assistant", "content": "SQL 쿼리 생성을 완료했습니다."}
        tool_message = {
            "role": "tool", 
            "content": result["output"],
            "name": "generate_sql_query",
            "tool_call_id": str(uuid4())
        }
        new_state["messages"] = state.messages + [assistant_message, tool_message]
        
        return new_state
        
    except Exception as e:
        # 오류 처리
        print(f"Error in query_generator_node: {str(e)}")
        traceback.print_exc()
        
        return {
            **sanitize_state_for_serialization(state),
            "error": {"message": str(e)},
            "status": "error"
        }


def data_loader_node(state: DBState) -> Dict[str, Any]:
    """
    데이터 로더 노드: 생성된 SQL 쿼리를 실행하여 데이터를 로드하고 처리합니다.
    """
    # 상태 확인
    if not state.generated_queries:
        return {**sanitize_state_for_serialization(state), "error": {"message": "생성된 쿼리가 없습니다."}, "status": "error"}
    
    generated_queries = state.generated_queries
    dbm = state.dbm
    
    try:
        # 데이터 로드
        df_dict = {}
        for table, query in generated_queries.items():
            try:
                df = dbm.select_from_table(query)
                df_dict[table] = df
                print(f"Successfully loaded data for {table}: {df.shape}")
            except Exception as query_error:
                print(f"Error loading data for {table}: {str(query_error)}")
                # 개별 쿼리 오류는 기록하되 전체 프로세스는 계속 진행

        # DataFrame이 하나도 로드되지 않은 경우
        if not df_dict:
            raise ValueError("모든 쿼리 실행 중 오류가 발생했습니다. 데이터를 불러올 수 없습니다.")

        # 상태 업데이트 (df_dict는 직렬화에서 제외)
        new_state = sanitize_state_for_serialization(state)
        
        # 메시지 추가
        assistant_message = {"role": "assistant", "content": "데이터 로딩을 완료했습니다."}
        tool_message = {
            "role": "tool", 
            "content": "데이터 로딩 완료",
            "name": "load_data",
            "tool_call_id": str(uuid4())
        }
        new_state["messages"] = state.messages + [assistant_message, tool_message]
        
        # DataFrame은 직접 반환용 딕셔너리에 추가
        return {
            **new_state,
            "df_dict": df_dict  # 직렬화에서 제외되지만 다음 노드로 전달하기 위해 포함
        }
        
    except Exception as e:
        # 오류 처리
        print(f"Error in data_loader_node: {str(e)}")
        traceback.print_exc()
        
        return {
            **sanitize_state_for_serialization(state),
            "error": {"message": str(e)},
            "status": "error"
        }


def python_executor_node(state: DBState) -> Dict[str, Any]:
    """
    파이썬 코드 실행 노드: 로드된 데이터를 파이썬 코드로 처리하여 사용자 질문에 맞는 답을 반환합니다.
    LangChain ReAct 에이전트를 사용하여 데이터 분석을 수행합니다.
    """
    # 상태 확인
    if not state.df_dict:
        return {**sanitize_state_for_serialization(state), "error": {"message": "데이터가 없습니다."}, "status": "error"}
    
    df_dict = state.df_dict
    question = state.question if state.question else ""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    
    try:
        # ReAct 에이전트 실행
        reaction_results = run_reaction_agent(question, df_dict, llm)
        
        # 결과로 상태 업데이트
        updated_state = update_state_with_reaction_results(state, reaction_results)
        
        # 원본 DataFrame들 추가 (직렬화 제외)
        if reaction_results.get("success", False):
            updated_state["df_dict"] = reaction_results.get("raw_df_dict", {})
            updated_state["result_df"] = reaction_results.get("raw_result_df")
        
        return updated_state
        
    except Exception as e:
        # 오류 처리
        print(f"Error in python_executor_node: {str(e)}")
        traceback.print_exc()
        
        # 가능하면 builtins에서 df_dict 제거
        try:
            if hasattr(builtins, 'df_dict'):
                del builtins.df_dict
        except:
            pass
            
        return {
            **sanitize_state_for_serialization(state),
            "error": {"message": str(e)},
            "status": "error"
        }


# 상태 라우터 함수들
def has_error(state: DBState) -> str:
    """오류가 있는지 확인"""
    if state.error:
        return "error"
    return "continue"

def has_selected_tables(state: DBState) -> str:
    """테이블이 선택되었는지 확인"""
    if state.selected_tables:
        return "has_tables"
    return "needs_tables"

def has_generated_queries(state: DBState) -> str:
    """쿼리가 생성되었는지 확인"""
    if state.generated_queries:
        return "has_queries"
    return "needs_queries"

def has_loaded_data(state: DBState) -> str:
    """데이터가 로드되었는지 확인"""
    if state.df_dict:
        return "has_data"
    return "needs_data"


# 그래프 생성 함수
def create_db_graph():
    """개선된 그래프를 생성합니다."""
    # 그래프 객체 생성
    workflow = StateGraph(DBState)
    
    # 노드 추가
    workflow.add_node("table_selector", table_selector_node)
    workflow.add_node("query_generator", query_generator_node)
    workflow.add_node("data_loader", data_loader_node)
    workflow.add_node("python_executor", python_executor_node)
    
    # 시작 노드 설정 - 이 부분이 중요!
    workflow.set_entry_point("table_selector")
    
    # 엣지 추가
    workflow.add_edge("table_selector", "query_generator")
    workflow.add_edge("query_generator", "data_loader")
    workflow.add_edge("data_loader", "python_executor")
    workflow.add_edge("python_executor", END)
    
    # 조건부 엣지 (오류 처리)
    workflow.add_conditional_edges(
        "table_selector",
        has_error,
        {
            "error": END,
            "continue": "query_generator"
        }
    )
    
    workflow.add_conditional_edges(
        "query_generator",
        has_error,
        {
            "error": END,
            "continue": "data_loader"
        }
    )
    
    workflow.add_conditional_edges(
        "data_loader",
        has_error,
        {
            "error": END,
            "continue": "python_executor"
        }
    )
    
    workflow.add_conditional_edges(
        "python_executor",
        has_error,
        {
            "error": END,
            "continue": END
        }
    )
    
    # 컴파일된 그래프 반환
    return workflow.compile()


# 프로세스 실행 함수
def process_query(
    question: str, 
    messages: List[Dict[str, Any]] = None, 
    conf = None, 
    dbm = None, 
    db = None
) -> Dict[str, Any]:
    """
    사용자 질문을 처리하여 답변을 생성합니다.
    모든 직렬화 문제를 처리합니다.
    """
    try:
        print(f"process_query 시작: question={question}")
        
        # messages가 None이면 기본값 설정
        if messages is None:
            messages = [{"role": "user", "content": question}]
            
        # 초기 상태 생성
        initial_state = DBState(
            question=question,
            messages=messages,
            conf=conf,
            dbm=dbm,
            db=db
        )
        
        print("초기 상태 생성 완료")
        
        # 그래프 생성 및 실행
        print("그래프 생성 시작")
        graph = create_db_graph()
        print("그래프 생성 완료, 실행 시작")
        result = graph.invoke(initial_state)
        print("그래프 실행 완료")
        
        # 결과 처리 및 반환
        print("결과 반환")
        return result
        
    except Exception as e:
        # 오류 처리
        print(f"Error in process_query: {str(e)}")
        print("\n상세 오류:")
        traceback.print_exc()
        
        # 직렬화 가능한 오류 결과 반환
        return {
            "status": "error",
            "error": {"message": str(e)},
            "messages": messages + [
                {"role": "assistant", "content": f"처리 중 오류가 발생했습니다: {str(e)}"}
            ]
        }

# main 실행 블록
if __name__ == "__main__":
    # matplotlib 폰트 설정 (한글 경고 해결)
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
    
    # 코드 실행
    conf = AgentConfig()
    dbm = DBManager(conf)
    db = dbm.get_db_connection()

    question = "3월 11일 원수 탁도의 평균을 시간대별로 알려줘."
    messages = [{"role": "user", "content": question}]
    
    try:
        print("프로세스 시작...")
        result = process_query(question, messages, conf, dbm, db)
        
        print("\n결과 요약:")
        print(f"상태: {result.get('status', 'unknown')}")
        if 'error' in result:
            print(f"오류: {result['error']}")
        print(f"선택된 테이블: {result.get('selected_tables', {})}")
        print(f"생성된 쿼리: {result.get('generated_queries', {})}")
        
        # df_dict는 직렬화에서 제외되었으므로 특별 처리
        if 'df_dict' in result:
            df_dict = result['df_dict']
            print(f"데이터프레임: {', '.join(df_dict.keys()) if df_dict else '없음'}")
        else:
            print("데이터프레임: 없음")
            
        print(f"파이썬 코드: {'있음' if result.get('python_code') else '없음'}")
        
        print("\n대화 내역:")
        for msg in result.get("messages", []):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            print(f"{role.upper()}: {content[:100]}..." if len(content) > 100 else f"{role.upper()}: {content}")

        print("프로세스 종료")
        
    except Exception as e:
        print(f"\n실행 중 오류 발생: {str(e)}")
        traceback.print_exc()
        
    print("끝")
