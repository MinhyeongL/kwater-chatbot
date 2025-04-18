"""
개선된 ReAct 에이전트 구현 모듈입니다.
직렬화 오류 처리와 에이전트 실행을 개선하였습니다.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import builtins
import traceback
from uuid import uuid4
import json
import re

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents.format_scratchpad import format_to_tool_messages
from langchain.schema.runnable import RunnablePassthrough
from langchain_experimental.tools import PythonREPLTool
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool, tool

from serializer_utils import sanitize_state_for_serialization, deep_sanitize_dict, safe_serialize_value
from tools import CustomPythonREPLTool, save_figure
from utils import (
    format_dataframes_info, extract_saved_images, 
    extract_intermediate_steps, extract_python_code,
    format_reaction_log
)
from prompts import Prompt


def setup_tools() -> List[BaseTool]:
    """ReAct 에이전트에서 사용할 도구들을 설정합니다."""
    python_tool = CustomPythonREPLTool()
    save_fig_tool = save_figure
    
    return [python_tool, save_fig_tool]


def create_react_prompt() -> PromptTemplate:
    """ReAct 에이전트의 프롬프트를 생성합니다."""
    return Prompt().react_agent_prompt()


def create_react_agent(llm: Any, tools: List[BaseTool], prompt: PromptTemplate) -> AgentExecutor:
    """ReAct 에이전트를 생성합니다."""
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=8,  # 최대 반복 횟수 제한
        early_stopping_method="generate",  # 더 이상 도구를 호출하지 않을 때 중지
        return_intermediate_steps=True  # 중간 단계 기록 반환
    )
    
    return agent_executor


def clear_global_namespace():
    """전역 네임스페이스에서 생성된 변수들을 정리합니다."""
    try:
        # 제거할 변수 목록
        variables_to_clear = [
            'df_dict', 'result_df', 'df', 'hourly_data', 
            'summary_df', 'stats_df', 'final_df', 
            'df_result', 'df_final'
        ]
        
        for var_name in variables_to_clear:
            if hasattr(builtins, var_name):
                delattr(builtins, var_name)
                print(f"Debug - Removed '{var_name}' from builtins")
                
    except Exception as e:
        print(f"Debug - Error clearing namespace: {str(e)}")


def collect_generated_dataframes(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """전역 네임스페이스에서 생성된 데이터프레임을 수집합니다."""
    try:
        # result_df 변수 확인
        if hasattr(builtins, 'result_df') and isinstance(builtins.result_df, pd.DataFrame):
            print("Debug - Found 'result_df' in builtins, copying to result")
            # 기존 데이터프레임은 유지하면서 결과 데이터프레임 추가
            df_dict['result_df'] = builtins.result_df.copy()
            
        # 다른 변수명으로 저장된 결과 확인
        for var_name in ['hourly_data', 'summary_df', 'stats_df', 'final_df', 'df_result', 'df_final']:
            if hasattr(builtins, var_name) and isinstance(getattr(builtins, var_name), pd.DataFrame):
                print(f"Debug - Found '{var_name}' in builtins, copying to result")
                df_dict[var_name] = getattr(builtins, var_name).copy()
        
        # 일반 'df' 변수 확인 (다른 결과가 없을 경우)
        if 'result_df' not in df_dict and hasattr(builtins, 'df') and isinstance(builtins.df, pd.DataFrame):
            print("Debug - Found 'df' in builtins, copying to result_df")
            df_dict['result_df'] = builtins.df.copy()
            
    except Exception as e:
        print(f"Debug - Error checking for dataframes: {str(e)}")
    
    return df_dict


def run_reaction_agent(question: str, df_dict: Dict[str, pd.DataFrame], llm: Any) -> Dict[str, Any]:
    """
    ReAct 에이전트를 실행하고 결과를 반환합니다.
    모든 직렬화 문제를 처리합니다.
    """
    try:
        # 1. 전역 네임스페이스에 데이터프레임 설정
        builtins.df_dict = df_dict
        
        # 2. 데이터프레임 정보 형식화
        dataframes_info = format_dataframes_info(df_dict)
        
        # 3. 도구 설정
        tools = setup_tools()
        
        # 4. 프롬프트 생성
        prompt = create_react_prompt()
        
        # 5. ReAct 에이전트 생성
        agent_executor = create_react_agent(llm, tools, prompt)
        
        # 6. 에이전트 실행
        result = agent_executor.invoke({
            "question": question,
            "dataframes_info": dataframes_info,
            "tools": "\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
            "tool_names": ", ".join([tool.name for tool in tools]),
            "agent_scratchpad": ""  # 빈 문자열로 시작
        })
        
        # 7. 결과 텍스트 추출
        final_text_result = result.get("output", "결과를 찾을 수 없습니다.")
        
        # 8. 저장된 이미지 추출 및 참조 추가
        image_references = extract_saved_images(result)
        if image_references:
            final_text_result += image_references
        
        # 9. 생성된 데이터프레임 수집
        updated_df_dict = collect_generated_dataframes(df_dict)
        
        # 10. 중간 단계 로그 추출
        intermediate_steps = extract_intermediate_steps(result)
        
        # 11. Python 코드 추출
        python_code = extract_python_code(intermediate_steps)
        
        # 12. 반응 로그 형식화
        react_log = format_reaction_log(intermediate_steps)
        
        # 13. 직렬화 가능한 결과 구성
        sanitized_result = {
            "final_result": final_text_result,
            "python_code": python_code,
            "react_log": react_log,
            "intermediate_steps": sanitize_safe_steps(intermediate_steps),
            "df_dict": "<직렬화 불가, 별도 필드로 반환>",
            "result_df": "<직렬화 불가, 별도 필드로 반환>" if "result_df" in updated_df_dict else None
        }
        
        # 14. 전역 네임스페이스 정리
        clear_global_namespace()
        
        # 15. 성공 결과 반환
        return {
            "success": True,
            "result": sanitized_result,
            "raw_df_dict": updated_df_dict,  # 원본 데이터프레임 딕셔너리
            "raw_result_df": updated_df_dict.get("result_df")  # 결과 데이터프레임
        }
        
    except Exception as e:
        # 오류 발생 시 정리 및 오류 반환
        print(f"Error in run_reaction_agent: {str(e)}")
        traceback.print_exc()
        
        # 모든 전역 변수 정리 시도
        try:
            clear_global_namespace()
        except:
            pass
            
        # 오류 결과 반환
        return {
            "success": False,
            "error": {
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        }


def sanitize_safe_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """중간 단계 기록을 안전하게 직렬화 가능한 형태로 변환합니다."""
    safe_steps = []
    
    for step in steps:
        safe_step = {
            "action": str(step.get("action", "")),
            "observation": str(step.get("observation", ""))
        }
        safe_steps.append(safe_step)
        
    return safe_steps


def update_state_with_reaction_results(state: Any, reaction_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    ReAct 에이전트 실행 결과로 상태를 업데이트합니다.
    직렬화 문제를 처리하고 안전하게 상태를 반환합니다.
    """
    try:
        # 성공 여부 확인
        if not reaction_results.get("success", False):
            # 오류 발생 시 오류 상태 반환
            error_message = reaction_results.get("error", {}).get("message", "Unknown error")
            
            # 직접 상태 필드 복사하여 재귀 방지
            if hasattr(state, 'model_dump'):
                sanitized_state = state.model_dump()
            elif isinstance(state, dict):
                sanitized_state = state.copy()
            else:
                sanitized_state = {}
                
            return {
                **sanitized_state,
                "status": "error",
                "error": {"message": error_message}
            }
            
        # 결과 데이터 추출
        result_data = reaction_results.get("result", {})
        final_result = result_data.get("final_result", "결과를 찾을 수 없습니다.")
        python_code = result_data.get("python_code", "")
        intermediate_steps = result_data.get("intermediate_steps", [])
        
        # 상태 데이터 추출 (직렬화 가능한 형태로)
        # 직접 상태 필드 복사하여 재귀 방지
        if hasattr(state, 'model_dump'):
            sanitized_state = state.model_dump()
        elif isinstance(state, dict):
            sanitized_state = state.copy()
        else:
            sanitized_state = {}
        
        # result_df가 있으면 상태 업데이트
        if reaction_results.get("raw_result_df") is not None:
            # result_df는 직렬화에서 제외되므로 별도로 처리
            print("Debug - Adding result_df to state")
        
        # 메시지 구성
        messages = sanitized_state.get("messages", [])
        assistant_message = {
            "role": "assistant", 
            "content": final_result
        }
        
        # Python 코드 도구 메시지 추가
        if python_code:
            tool_message = {
                "role": "tool", 
                "content": "Python 코드 실행 완료",
                "name": "python_executor",
                "tool_call_id": str(uuid4())
            }
            messages.append(assistant_message)
            messages.append(tool_message)
        else:
            messages.append(assistant_message)
        
        # 최종 상태 구성
        updated_state = {
            **sanitized_state,
            "messages": messages,
            "python_code": python_code,
            "final_result": final_result,
            "intermediate_steps": intermediate_steps,
            "status": "completed"
        }
        
        return updated_state
        
    except Exception as e:
        # 오류 발생 시 기본 정보만 반환
        print(f"Error updating state: {str(e)}")
        # 재귀 방지를 위해 traceback 출력 제거
        # traceback.print_exc()
        
        return {
            "status": "error",
            "error": {"message": f"Error updating state: {type(e).__name__}"}
        } 