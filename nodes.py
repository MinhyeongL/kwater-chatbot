import operator
from typing import TypedDict, List, Dict, Any, Optional, Annotated, Literal
import pandas as pd
import json

from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import create_tool_calling_agent, create_react_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from uuid import uuid4

from utils import *
from tools import *
from agents import run_reaction_agent, update_state_with_reaction_results
from prompts import Prompt
from states import DBState
from serializer_utils import sanitize_state_for_serialization, sanitize_intermediate_steps

import builtins
import traceback


import os
os.environ["PYDANTIC_STRICT_SCHEMA_VALIDATION"] = "False"

# LLM 모델 설정
MODEL_NAME = "gpt-4o"  # 또는 다른 모델 이름 사용


def db_supervisor_node(state: DBState) -> Dict[str, Any]:
    """
    데이터베이스 관리자 노드
    1. state의 status가 running이면 다음 노드 선택
    2. state의 status가 completed이면 최종 답변 생성
    3. state의 status가 error면 이전 노드에서 에러 해결
    """
    # 데이터베이스 연결 확인
    if not state.db:
        return {**sanitize_state_for_serialization(state), "error": {"message": "데이터베이스 연결이 없습니다."}, "status": "error"}
    
    try:
        # Supervisor LLM 생성
        llm_for_supervisor = ChatOpenAI(model=MODEL_NAME, temperature=0)
        
        # Supervisor 프롬프트 생성
        supervisor_prompt = Prompt().db_supervisor_prompt()

        # 현재 상태 정보 수집
        status_info = {
            "has_tables": bool(state.selected_tables),
            "has_queries": bool(state.generated_queries),
            "has_data": bool(state.df_dict),
            "has_python_code": bool(state.python_code),
            "has_result": bool(state.final_result),
            "status": state.status,
            "question": state.question if state.question else "",
            "last_node": state.last_node if hasattr(state, "last_node") else None,
            "error_message": state.error.get("message", "") if hasattr(state, "error") and state.error else ""
        }
        
        # Agent 도구 정의 (향후 확장을 위한 준비)
        """
        # Agent를 사용할 경우 아래와 같이 도구를 정의하고 사용할 수 있습니다
        tools = [
            Tool(
                name="decide_next_node",
                func=lambda input_str: {"result": "next_node_decided", "next_node": input_str},
                description="현재 상태를 분석하여 다음에 실행할 노드를 결정합니다."
            ),
            Tool(
                name="generate_final_answer",
                func=lambda input_str: {"result": "answer_generated", "answer": input_str},
                description="최종 결과를 분석하여 사용자 질문에 대한 최종 답변을 생성합니다."
            ),
            Tool(
                name="analyze_error",
                func=lambda input_str: {"result": "error_analyzed", "analysis": input_str},
                description="오류의 원인을 분석하고 해결 방안을 제시합니다."
            )
        ]
        
        # Agent 생성 및 실행
        agent = create_tool_calling_agent(llm_for_supervisor, tools, supervisor_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Agent 실행
        agent_result = agent_executor.invoke({
            "status_info": json.dumps(status_info, ensure_ascii=False, indent=2)
        })
        
        # Agent 결과 처리
        action_response = agent_result["output"]
        """
        
        # 현재는 체인 방식 사용
        supervisor_chain = supervisor_prompt | llm_for_supervisor | StrOutputParser()
        action_response = supervisor_chain.invoke({
            "status_info": json.dumps(status_info, ensure_ascii=False, indent=2)
        })
        
        # 응답 파싱
        try:
            action_data = json_parse(action_response)
            action_type = action_data.get("action")
            
            # 상태에 따른 처리
            if action_type == "next_node":
                # 다음 노드 선택 (success 상태)
                next_node = action_data.get("next_node")
                reason = action_data.get("reason", "")
                
                # data_loader 또는 python_code_executor로의 결정은 무시 (직접 연결되므로)
                if next_node == "data_loader":
                    if not state.df_dict:
                        next_node = "query_generator"  # query_generator가 data_loader로 연결됨
                    else:
                        next_node = "python_code_generator"  # 이미 데이터가 있으면 python_code_generator로
                        
                if next_node == "python_code_executor":
                    if not state.python_code:
                        next_node = "python_code_generator"  # python_code_generator가 python_code_executor로 연결됨
                
                # 현재 노드를 last_node로 저장 (오류 발생 시 돌아오기 위해)
                new_state = sanitize_state_for_serialization(state)
                new_state["next"] = next_node
                new_state["status"] = "running"
                new_state["last_node"] = "supervisor"  # 현재 노드 저장
                
                # 중간 단계 기록
                if not new_state.get("intermediate_steps"):
                    new_state["intermediate_steps"] = []
                
                new_state["intermediate_steps"].append({
                    "node": "supervisor",
                    "action": "decide_next_node",
                    "result": next_node,
                    "reason": reason
                })
                
                return new_state
                
            elif action_type == "generate_answer":
                # 최종 답변 생성 (completed 상태)
                answer = action_data.get("answer", "")
                reason = action_data.get("reason", "")
                
                # 상태 업데이트
                new_state = sanitize_state_for_serialization(state)
                new_state["status"] = "completed"
                new_state["completed"] = True
                new_state["final_answer"] = answer  # 최종 답변 저장
                
                # 최종 답변 메시지 추가
                assistant_message = {
                    "role": "assistant", 
                    "content": answer
                }
                
                if not new_state.get("messages"):
                    new_state["messages"] = []
                
                new_state["messages"] = state.messages + [assistant_message]
                
                # 중간 단계 기록
                if not new_state.get("intermediate_steps"):
                    new_state["intermediate_steps"] = []
                
                new_state["intermediate_steps"].append({
                    "node": "supervisor",
                    "action": "generate_answer",
                    "result": "final_answer_generated",
                    "reason": reason
                })
                
                return new_state
                
            elif action_type == "fix_error":
                # 에러 해결 (error 상태)
                error_node = action_data.get("error_node")
                error_analysis = action_data.get("error_analysis", "")
                solution = action_data.get("solution", "")
                retry_node = action_data.get("retry_node")
                
                # 상태 업데이트
                new_state = sanitize_state_for_serialization(state)
                new_state["status"] = "retry"  # 재시도 상태로 변경
                new_state["next"] = retry_node  # 재시도할 노드 설정
                
                # 에러 해결 정보 추가
                new_state["error_resolution"] = {
                    "error_node": error_node,
                    "error_analysis": error_analysis,
                    "solution": solution
                }
                
                # 오류 정보 초기화 (재시도를 위해)
                if "error" in new_state:
                    del new_state["error"]
                
                # 에러 해결 메시지 추가 (사용자에게 보여줄 정보)
                assistant_message = {
                    "role": "assistant", 
                    "content": f"오류가 발생했습니다: {error_analysis}\n\n해결 방안: {solution}\n\n다시 시도하겠습니다."
                }
                
                if not new_state.get("messages"):
                    new_state["messages"] = []
                
                new_state["messages"] = state.messages + [assistant_message]
                
                # 중간 단계 기록
                if not new_state.get("intermediate_steps"):
                    new_state["intermediate_steps"] = []
                
                new_state["intermediate_steps"].append({
                    "node": "supervisor",
                    "action": "fix_error",
                    "result": f"retry_{retry_node}",
                    "reason": error_analysis
                })
                
                return new_state
            
            else:
                # 알 수 없는 액션 타입
                raise ValueError(f"알 수 없는 액션 타입: {action_type}")
                
        except Exception as parse_error:
            # 파싱 오류 처리
            print(f"Error parsing supervisor response: {str(parse_error)}")
            print(f"Raw response: {action_response}")
            
            # 기본 라우팅 로직으로 폴백
            next_node = None
            reason = "자동 결정 (파싱 오류)"
            
            # 상태에 따른 기본 라우팅
            if state.status == "error":
                # 오류 상태 처리
                if hasattr(state, "last_node") and state.last_node:
                    next_node = state.last_node  # 이전 노드로 돌아가기
                else:
                    # 오류 발생했지만 이전 노드가 없는 경우 기본 라우팅
                    if not state.selected_tables:
                        next_node = "table_selector"
                    elif not state.generated_queries:
                        next_node = "query_generator"
                    elif state.df_dict and not state.python_code:
                        next_node = "python_code_generator"
            else:
                # 일반 상태 처리
                if not state.selected_tables:
                    next_node = "table_selector"
                elif not state.generated_queries:
                    next_node = "query_generator"
                elif state.df_dict and not state.python_code and not state.final_result:
                    next_node = "python_code_generator"
                elif state.final_result:
                    # 이미 결과가 있으면 완료 처리
                    next_node = None
                    
            # 상태 업데이트
            new_state = sanitize_state_for_serialization(state)
            new_state["next"] = next_node
            new_state["status"] = "running" if next_node else "completed"
            new_state["last_node"] = "supervisor"  # 현재 노드 저장
            
            # 완료 여부 확인
            if state.final_result and not next_node:
                new_state["completed"] = True
            
            # 중간 단계 기록
            if not new_state.get("intermediate_steps"):
                new_state["intermediate_steps"] = []
            
            new_state["intermediate_steps"].append({
                "node": "supervisor",
                "action": "decide_next_node",
                "result": next_node,
                "reason": reason
            })
            
            return new_state
    
    except Exception as e:
        # 일반 오류 처리
        print(f"Error in supervisor_node: {str(e)}")
        traceback.print_exc()
        
        # 기본 라우팅 로직으로 폴백
        next_node = None
        reason = f"오류 발생: {str(e)}"
        
        # 상태 확인 및 기본 라우팅
        if not state.selected_tables:
            next_node = "table_selector"
        elif not state.generated_queries:
            next_node = "query_generator"
        elif state.df_dict and not state.python_code and not state.final_result:
            next_node = "python_code_generator"
        
        return {
            **sanitize_state_for_serialization(state),
            "next": next_node,
            "status": "running",
            "error_info": {"message": str(e)}  # 오류 정보 추가
        }


def table_selector_node(state: DBState) -> Dict[str, Any]:
    """
    테이블 선택 노드: 사용자 쿼리를 분석하여 관련된 데이터베이스 테이블을 선택합니다.
    """
    # 메시지에서 질문 추출
    if state.messages:
        last_message = state.messages[-1]
        question = last_message.get("content", "")
    else:
        return {**sanitize_state_for_serialization(state), "error": {"message": "메시지가 없습니다."}, "status": "error"}
    
    try:
        # 도구 정의
        tools = [select_table]
        
        # 프롬프트 생성
        table_selector_prompt = Prompt().table_selector_prompt()
        
        llm_for_agent = ChatOpenAI(model=MODEL_NAME, temperature=0)

        # 에이전트 생성
        agent = create_tool_calling_agent(llm_for_agent, tools=tools, prompt=table_selector_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            verbose=True
        )
        
        # 메시지 형식 변환
        messages_for_agent = []
        for msg in state.messages:
            if msg.get("role") == "user" or msg.get("role") == "human":
                messages_for_agent.append({"role": "user", "content": msg.get("content", "")})
            elif msg.get("role") == "assistant" or msg.get("role") == "ai":
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
        new_state["question"] = question  # 현재 질문 저장
        new_state["last_node"] = "table_selector"  # 현재 노드 저장
        
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
        return {
            **sanitize_state_for_serialization(state),
            "error": {"message": str(e)},
            "status": "error",
            "last_node": "table_selector"  # 오류 발생 노드 저장
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
        
        generated_queries = {}
        for table_name in selected_tables.keys():
            sql_query = sql_query_generator_chain.invoke({
                        "schema_info": schema_info,
                        "question": question,
                        "table_name": table_name
                    })
            generated_queries[table_name] = sql_query
            
        # 상태 업데이트
        new_state = sanitize_state_for_serialization(state)
        new_state["generated_queries"] = generated_queries
        new_state["last_node"] = "query_generator"  # 현재 노드 저장
        
        # 메시지 추가
        assistant_message = {"role": "assistant", "content": "SQL 쿼리를 생성했습니다."}
        new_state["messages"] = state.messages + [assistant_message]
        
        return new_state
        
    except Exception as e:
        # 오류 처리
        print(f"Error in query_generator_node: {str(e)}")
        traceback.print_exc()

        return {
            **sanitize_state_for_serialization(state),
            "error": {"message": str(e)},
            "status": "error",
            "last_node": "query_generator"  # 오류 발생 노드 저장
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
        builtins.dbm = dbm
        
        # 데이터 로드
        df_dict = load_data(generated_queries, dbm)
        if df_dict:
            for name, df in df_dict.items():
                print(f"Successfully loaded data for {name}: {df.shape}")
        else:
            raise ValueError("모든 쿼리 실행 중 오류가 발생했습니다. 데이터를 불러올 수 없습니다.")

        # 상태 업데이트
        new_state = sanitize_state_for_serialization(state)
        new_state["df_dict"] = df_dict
        new_state["last_node"] = "data_loader"  # 현재 노드 저장

        # 메시지 추가
        assistant_message = {"role": "assistant", "content": "데이터 로딩을 완료했습니다."}
        tool_message = {
            "role": "tool", 
            "content": "데이터 로딩 완료",
            "name": "load_data",
            "tool_call_id": str(uuid4())
        }
        new_state["messages"] = state.messages + [assistant_message, tool_message]
        
        return new_state
        
    except Exception as e:
        # 오류 처리
        print(f"Error in data_loader_node: {str(e)}")
        traceback.print_exc()

        return {
            **sanitize_state_for_serialization(state),
            "error": {"message": str(e)},
            "status": "error",
            "last_node": "data_loader"  # 오류 발생 노드 저장
        }


def python_code_generator_node(state: DBState) -> Dict[str, Any]:
    """
    파이썬 코드 생성 노드: 사용자 질문에 맞는 파이썬 코드를 생성합니다.
    """
    # 상태 확인
    if not state.df_dict:
        return {**sanitize_state_for_serialization(state), "error": {"message": "데이터가 로드되지 않았습니다."}, "status": "error"}
    
    question = state.question if state.question else ""
    df_dict = state.df_dict
    
    try:
        # 데이터프레임 정보 형식화
        dataframes_info = format_dataframes_info(df_dict)
        
        # 이전 실행에서 피드백 정보가 있는지 확인
        code_feedback_str = ""
        code_feedback_data = getattr(state, "code_feedback", None)
        if code_feedback_data:
            code_feedback_str = f"""
            이전 코드 실행에서 다음 문제가 발생했습니다:
            문제: {code_feedback_data.get('message', '알 수 없는 오류')}
            제안: {code_feedback_data.get('suggestion', '코드를 수정하세요')}
            """
        
        # 프롬프트 생성
        code_generator_prompt = Prompt().python_code_generator_prompt()
        
        # LLM 설정
        llm_for_code = ChatOpenAI(model=MODEL_NAME, temperature=0)
        
        # 코드 생성 체인 구성
        code_chain = (
            code_generator_prompt 
            | llm_for_code 
            | StrOutputParser()
        )
        
        # 코드 생성 실행
        generated_code = code_chain.invoke({
            "question": question,
            "dataframes_info": dataframes_info,
            "code_feedback": code_feedback_str
        })
        
        # 코드 추출 및 정제
        cleaned_code = extract_python_code(generated_code)
        
        # 상태 업데이트
        new_state = sanitize_state_for_serialization(state)
        new_state["python_code"] = cleaned_code
        new_state["last_node"] = "python_code_generator"  # 현재 노드 저장
        
        # 피드백 정보 제거 (새로운 코드를 생성했으므로)
        if "code_feedback" in new_state:
            del new_state["code_feedback"]
        
        # 메시지 추가
        assistant_message = {
            "role": "assistant", 
            "content": "데이터 분석을 위한 파이썬 코드를 생성했습니다."
        }
        tool_message = {
            "role": "tool", 
            "content": cleaned_code,
            "name": "python_code_generator",
            "tool_call_id": str(uuid4())
        }
        new_state["messages"] = state.messages + [assistant_message, tool_message]
        
        return new_state
        
    except Exception as e:
        # 오류 처리
        print(f"Error in python_code_generator_node: {str(e)}")
        traceback.print_exc()
        return {
            **sanitize_state_for_serialization(state),
            "error": {"message": str(e)},
            "status": "error",
            "last_node": "python_code_generator"  # 오류 발생 노드 저장
        }


def python_code_executor_node(state: DBState) -> Dict[str, Any]:
    """
    파이썬 코드 실행 노드: 생성된 파이썬 코드를 실행합니다.
    """
    # 상태 확인
    if not state.python_code:
        return {**sanitize_state_for_serialization(state), "error": {"message": "실행할 코드가 없습니다."}, "status": "error"}
    
    try:
        # 코드 실행 준비
        df_dict = state.df_dict
        code = state.python_code
        
        # 전역 네임스페이스에 데이터프레임 설정
        builtins.df_dict = df_dict
        
        local_vars = {
            "df_dict": df_dict
        }

        print(f"Debug - Executing code: {code[:200]}...")
        exec(code, globals(), local_vars)

        result_df = local_vars.get('result_df', None)
        print(f"Debug - Found result_df in local_vars: {type(result_df)}")
        
        
        # 결과 텍스트 생성
        if result_df is None:
            error_message = "result_df 변수가 생성되지 않았습니다. 파이썬 코드를 다시 확인해주세요."
            # 피드백 정보 추가
            return {
                **sanitize_state_for_serialization(state),
                "error": {"message": error_message},
                "status": "error",
                "last_node": "python_code_executor",  # 오류 발생 노드 저장
                "code_feedback": {
                    "issue": "missing_result_df",
                    "message": error_message,
                    "suggestion": "코드 마지막에 반드시 'result_df = ...' 형태로 결과를 저장해야 합니다."
                }
            }
        elif isinstance(result_df, pd.DataFrame):
            # 빈 DataFrame인지 확인
            if result_df.empty:
                error_message = "분석 결과가 비어 있습니다. 파이썬 코드를 다시 확인해주세요."
                return {
                    **sanitize_state_for_serialization(state),
                    "error": {"message": error_message},
                    "status": "error",
                    "last_node": "python_code_executor",  # 오류 발생 노드 저장
                    "code_feedback": {
                        "issue": "empty_result_df",
                        "message": error_message,
                        "suggestion": "데이터 필터링 조건을 확인하고, 결과가 비어있지 않도록 코드를 수정하세요."
                    }
                }
                
        # 이미지 처리 여부 확인하고 이미지 데이터 캡처
        import matplotlib.pyplot as plt
        import io
        import base64
        
        has_visualization = False
        result_image = None
        
        if plt.get_fignums():  # 열린 figure가 있는지 확인
            has_visualization = True
            
            try:
                # 현재 figure를 바이트로 저장
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                
                # 바이트 데이터를 base64로 인코딩
                result_image = base64.b64encode(buf.read()).decode('utf-8')
                print(f"Debug - Visualization data captured successfully")
            except Exception as img_error:
                print(f"Warning - Failed to capture visualization data: {str(img_error)}")
            finally:
                # 메모리 관리를 위해 모든 figure 닫기
                plt.close('all')
        
        # 상태 업데이트
        new_state = sanitize_state_for_serialization(state)
        new_state["result_df"] = result_df
        new_state["has_visualization"] = has_visualization
        if has_visualization:
            new_state["result_image"] = result_image
        new_state["status"] = "completed"  # 실행 완료 상태로 설정
        new_state["completed"] = True      # 전체 프로세스 완료 표시
        new_state["last_node"] = "python_code_executor"  # 현재 노드 저장
        
        # 메시지 추가
        assistant_message = {
            "role": "assistant", 
            "content": "코드 실행이 완료되었습니다."  # 단순 실행 완료 메시지만 추가
        }
        tool_message = {
            "role": "tool", 
            "content": "코드 실행 완료",
            "name": "python_code_executor",
            "tool_call_id": str(uuid4())
        }
        new_state["messages"] = state.messages + [assistant_message, tool_message]
        
        # 전역 네임스페이스 정리
        if hasattr(builtins, 'df_dict'):
            delattr(builtins, 'df_dict')
        if hasattr(builtins, 'result_df'):
            delattr(builtins, 'result_df')
            
        return new_state
        
    except Exception as e:
        # 오류 처리
        print(f"Error in python_code_executor_node: {str(e)}")
        traceback.print_exc()
        
        # 전역 네임스페이스 정리
        if hasattr(builtins, 'df_dict'):
            delattr(builtins, 'df_dict')
        if hasattr(builtins, 'result_df'):
            delattr(builtins, 'result_df')
            
        return {
            **sanitize_state_for_serialization(state),
            "error": {"message": f"코드 실행 중 오류 발생: {str(e)}"},
            "status": "error",
            "last_node": "python_code_executor"  # 오류 발생 노드 저장
        }


# 기존 함수 주석 처리 (참고용)
"""
def python_executor_node(state: DBState) -> Dict[str, Any]:
    # 상태 확인
    if not state.df_dict:
        return {**sanitize_state_for_serialization(state), "error": {"message": "데이터가 없습니다."}, "status": "error"}
    
    df_dict = state.df_dict
    question = state.question if state.question else ""
    llm_for_agent = ChatOpenAI(model=MODEL_NAME, temperature=0)

    try: # 여기를 고쳐야 할 듯... 
        # ReAct 에이전트 실행
        reaction_results = run_reaction_agent(question, df_dict, llm_for_agent)
        
        # 결과로 상태 업데이트
        updated_state = update_state_with_reaction_results(state, reaction_results)
        
        # 원본 DataFrame들 추가 (직렬화 제외)
        if reaction_results.get("running", False):
            updated_state["df_dict"] = reaction_results.get("raw_df_dict", {})
            updated_state["result_df"] = reaction_results.get("raw_result_df")
            
            # 분석 완료 메시지 추가
            updated_state["completed"] = True
        
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
"""
        