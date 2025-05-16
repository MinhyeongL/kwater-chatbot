from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import json
import re
import builtins
import traceback



AVAILABLE_TABLES = {
    "TB_C_RT": "실시간 측정 데이터 테이블. 센서의 실시간 측정값을 저장합니다.",
    "TB_AI_C_RT": "AI 분석 결과 테이블. AI가 분석한 예측값과 결과를 저장합니다.",
    "TB_AI_C_CTR": "AI 제어 결과 테이블. AI의 제어 명령과 결과를 저장합니다.",
    "TB_AI_C_INIT": "AI 알고리즘의 초기 설정 테이블. 상한, 하한 및 초기 설정 값을 저장합니다.",
    "TB_TAG_MNG": "태그 관리 테이블. 시스템에서 사용하는 태그(센서 등)의 메타데이터를 저장합니다."
}


def extract_tables(llm_response: str) -> List[str]:
    """LLM 응답에서 테이블 이름 목록을 추출합니다."""
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


def json_parse(json_string):
    """더 강력한 JSON 파싱 함수"""


    # 1. 코드 블록 마크다운 제거
    clean_str = re.sub(r'```(?:json)?|```', '', json_string).strip()

    # 2. 중괄호 기준으로 JSON 부분 추출
    match = re.search(r'\{.*\}', clean_str, re.DOTALL)
    if match:
        json_part = match.group(0)
    else:
        json_part = clean_str

    try:
        # 3. 기본 파싱 시도
        return json.loads(json_part)
    except json.JSONDecodeError:
        try:
            # 4. SQL 쿼리 형식 처리: 파이썬 딕셔너리로 변환 시도
            # TB_NAME: QUERY1, TB_NAME2: QUERY2 형식 처리
            result = {}
            
            # 4.1 콜론을 기준으로 테이블 이름과 쿼리 분리
            pattern = r'([A-Za-z0-9_]+)\s*:\s*(.*?)(?=,\s*[A-Za-z0-9_]+\s*:|$)'
            matches = re.findall(pattern, json_part, re.DOTALL)
            
            if matches:
                for table, query in matches:
                    result[table.strip()] = query.strip()
                
                if result:
                    return result
        except:
            pass
                
        # 5. 줄 단위로 정리 시도
        lines = json_part.split('\n')
        clean_lines = []

        for line in lines:
            # 주석이나 비정상적인 줄 제거
            if '//' in line:
                line = line[:line.index('//')]
            clean_lines.append(line.strip())

        clean_json = ' '.join(clean_lines)

        try:
            # 6. 다시 파싱 시도
            return json.loads(clean_json)
        except:
            # 7. JSON 형식 수정 시도 (작은따옴표를 큰따옴표로 변경, 등)
            try:
                fixed_json = re.sub(r'(\w+):', r'"\1":', clean_json)
                fixed_json = re.sub(r"'([^']*)'", r'"\1"', fixed_json)
                return json.loads(fixed_json)
            except:
                pass
                
            # 8. 최후의 수단: 정규식을 사용한 키-값 추출
            # 8.1 키:값 패턴 (쌍따옴표가 있는 경우)
            result = {}
            pattern = r'"([^"]*)"\s*:\s*"([^"]*)"'
            matches = re.findall(pattern, json_part)
            
            for key, value in matches:
                result[key] = value
                
            # 8.2 키:값 패턴 (쌍따옴표가 없는 경우)
            if not result:
                pattern = r'([A-Za-z0-9_]+)\s*:\s*([^,]+)'
                matches = re.findall(pattern, json_part)
                
                for key, value in matches:
                    result[key.strip()] = value.strip()
                    
            return result


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

        # 딕셔너리 변환 시도
        try:
            return json.loads(value)
        except:
            pass

    # 변환 실패 시 원래 값 반환
    return value


# 바이트 문자열을 파싱하여 딕셔너리로 변환하는 함수
def parse_byte_string(byte_string):
    if isinstance(byte_string, bytes):
        decoded_string = byte_string.decode('utf-8')
        return json.loads(decoded_string)
    return byte_string

# 문자열을 적절한 데이터 타입으로 변환하는 함수
def convert_str_to_value(value):
    if value is None:
        return value
    
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
    
    # 변환 실패 시 원래 문자열 반환
    return value

# 재귀적으로 데이터를 처리하는 함수
def process_ai_rt(data, path=()):
    # data = parse_byte_string(data)
    # data = json.loads(data)
    data = convert_value(data)
    if isinstance(data, dict):
        for key, value in data.items():
            if key.startswith(('STG_', 'SER_', 'LOC_', 'STP_')):
                key = int(key[4:])
            elif key.startswith(('STG', 'SER', 'LOC', 'STP')):
                key = int(key[3:])
            yield from process_ai_rt(value, path + (key,))
    else:
        data = convert_str_to_value(data.split(";")[0] if data else data)
        path = path + (0,) * (5 - len(path))
        yield path, data


def process_schema_for_prompt(schema_text):
    """테이블별로 스키마 정보 처리"""
    return schema_text.split("/*")[0].strip()


def format_tables_for_prompt(tables_dict):
    if not tables_dict:
        return "없음"
    
    formatted_lines = []
    for table_name, description in tables_dict.items():
        formatted_lines.append(f"- {table_name}: {description}")
    
    return "\n".join(formatted_lines)

def format_dataframes_info(df_dict):
    result = []
    for table_name, df in df_dict.items():
        df_info = f"- {table_name}: {df.shape[0]}행 x {df.shape[1]}열"
        
        # 컬럼 정보 추가
        columns_info = ", ".join(df.columns[:10].tolist())
        if len(df.columns) > 10:
            columns_info += f" 외 {len(df.columns) - 10}개 컬럼"
        
        df_info += f"\n  컬럼: {columns_info}"
        
        # 데이터 타입 정보 추가
        df_info += f"\n  데이터 타입: {df.dtypes.value_counts().to_dict()}"
        
        # 샘플 데이터 추가
        if not df.empty:
            df_info += f"\n  샘플 데이터:\n{df.head(3).to_string()}"
        
        result.append(df_info)
    
    return "\n\n".join(result)

def collect_generated_dataframes(df_dict):
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

def extract_saved_images(result):
    """중간 단계 로그에서 저장된， 이미지 파일 경로를 추출합니다."""
    
    
    image_references = ""
    saved_images = re.findall(r'이미지가 (.*?)에 저장되었습니다', str(result.get("intermediate_steps", "")))
    
    if saved_images:
        image_references = "\n\n### 생성된 이미지\n"
        for img_path in saved_images:
            # 경로에서 따옴표 제거
            clean_path = img_path.strip("'\"")
            image_references += f"![{clean_path}]({clean_path})\n"
            
    return image_references

def extract_intermediate_steps(result):
    """ReAct 에이전트의 중간 단계 로그를 추출합니다."""
    new_intermediate_steps = []
    
    if "intermediate_steps" in result:
        for step in result["intermediate_steps"]:
            if len(step) >= 2:
                action = step[0]
                observation = step[1]
                new_intermediate_steps.append({
                    "action": str(action),
                    "observation": str(observation)
                })
                
    return new_intermediate_steps

def extract_python_code(input_data):
    """
    파이썬 코드를 추출합니다.
    input_data는 문자열이거나 에이전트의 intermediate_steps일 수 있습니다.
    """
    # 문자열 입력인 경우 (생성된 코드에서 직접 추출)
    if isinstance(input_data, str):
        # 코드 블록 추출 (```python ... ``` 패턴)
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', input_data, re.DOTALL)
        if code_blocks:
            return '\n\n'.join(code_blocks)
            
        # 백틱 없이 전체가 코드인 경우
        if input_data.strip().startswith('import ') or input_data.strip().startswith('# '):
            return input_data.strip()
            
        # 기타 경우: 그냥 입력을 반환
        return input_data
    
    # intermediate_steps 리스트인 경우 (ReAct 에이전트의 중간 단계에서 추출)
    python_code = ""
    for step in input_data:
        if "python_repl_tool" in str(step["action"]):
            # 디버깅 출력 추가
            print(f"Debug - Action found: {step['action']}")
            
            try:
                # AgentAction 객체 직접 액세스 시도 (직렬화 전)
                if hasattr(step["action"], "tool_input") and step["action"].tool == "python_repl_tool":
                    python_code += str(step["action"].tool_input) + "\n\n"
                    print(f"Debug - Direct access: {str(step['action'].tool_input)[:50]}...")
                    continue
                    
                # Action Input 형식 구분 (Action: python_repl_tool\nAction Input: ...)
                action_str = str(step["action"])
                if "Action Input:" in action_str:
                    code_part = action_str.split("Action Input:", 1)[1].strip()
                    python_code += code_part + "\n\n"
                    print(f"Debug - Extracted code from Action Input: {code_part[:50]}...")
                    continue
                    
                # 여러 가지 패턴으로 시도
                # 패턴 1: input='...'
                tool_input_match = re.search(r"input='([^']*)'", str(step["action"]))
                if tool_input_match:
                    extracted_code = tool_input_match.group(1)
                    python_code += extracted_code + "\n\n"
                    print(f"Debug - Extracted code (pattern 1): {extracted_code[:50]}...")
                    continue
                    
                # 패턴 2: tool_input=...
                tool_input_match = re.search(r"tool_input=\"?(.*?)\"?,", str(step["action"]))
                if tool_input_match:
                    extracted_code = tool_input_match.group(1)
                    python_code += extracted_code + "\n\n"
                    print(f"Debug - Extracted code (pattern 2): {extracted_code[:50]}...")
                    continue
                    
                # 패턴 3: tool_input 단어 이후의 모든 텍스트
                tool_input_match = re.search(r"tool_input=(.*?)(?:,|\))", str(step["action"]))
                if tool_input_match:
                    extracted_code = tool_input_match.group(1).strip("'\"")
                    python_code += extracted_code + "\n\n"
                    print(f"Debug - Extracted code (pattern 3): {extracted_code[:50]}...")
                    continue
                    
                # 패턴 4: 따옴표로 된 모든 코드 섹션 찾기
                code_sections = re.findall(r"'([^']*)'", str(step["action"]))
                if code_sections:
                    for code in code_sections:
                        if len(code) > 10:  # 충분히 긴 코드 섹션만 포함
                            python_code += code + "\n\n"
                            print(f"Debug - Extracted code (pattern 4): {code[:50]}...")
                        continue
                        
                # 백업: 전체 액션 로그
                python_code += f"# 추출 실패한 코드 액션:\n# {str(step['action'])}\n\n"
                print(f"Debug - Failed to extract code from: {str(step['action'])[:100]}...")
                
            except Exception as e:
                print(f"Debug - Error extracting code: {str(e)}")
                python_code += f"# 코드 추출 중 오류 발생: {str(e)}\n# {str(step['action'])}\n\n"
                
    # 디버그 로그 추가
    print(f"Debug - Total extracted Python code length: {len(python_code)}")
    
    return python_code

def format_reaction_log(intermediate_steps):
    """중간 단계 로그를 사용자 친화적인 형식으로 변환합니다."""
    react_log = ""
    
    if intermediate_steps:
        react_log = "\n\n### 분석 과정\n"
        for i, step in enumerate(intermediate_steps):
            react_log += f"**단계 {i+1}**\n"
            react_log += f"- 실행: {step['action']}\n"
            react_log += f"- 결과: {step['observation']}\n\n"
            
    return react_log

def update_state(state, final_result, react_log, python_code, intermediate_steps):
    """상태 객체를 업데이트합니다."""
    # 결과 메시지 추가
    assistant_message = {
        "role": "assistant", 
        "content": final_result + (react_log if len(react_log) < 1000 else "")  # 로그가 너무 길면 생략
    }
    
    # 디버그 출력 추가
    print(f"Debug - Final Result Length: {len(final_result)}")
    print(f"Debug - React Log Length: {len(react_log)}")
    print(f"Debug - Assistant Message Content Length: {len(assistant_message['content'])}")

    # 상태 복사 및 업데이트
    new_state = {**state.model_dump()}
    
    # python_code와 중간 단계 업데이트
    new_state["python_code"] = python_code  # 추출된 Python 코드 저장
    new_state["intermediate_steps"] = intermediate_steps  # ReAct 단계 저장
    
    # 메시지 추가
    if "messages" in new_state:
        print(f"Debug - Before update: messages count = {len(new_state['messages'])}")
        new_state["messages"] = state.messages + [assistant_message]
        print(f"Debug - After update: messages count = {len(new_state['messages'])}")
    else:
        print("Debug - Creating new messages list")
        new_state["messages"] = [assistant_message]
    
    # 상태 확인
    print(f"Debug - Final State Keys: {new_state.keys()}")
    print(f"Debug - Python Code in State: {'Yes' if new_state.get('python_code') else 'No'}")
    print(f"Debug - Python Code Length: {len(new_state.get('python_code', ''))}")
    print(f"Debug - Messages in State: {'Yes' if new_state.get('messages') else 'No'}")
    print(f"Debug - Messages Count: {len(new_state.get('messages', []))}")
    
    return new_state

def handle_executor_error(state, e):
    """파이썬 실행 노드에서 발생한 오류를 처리합니다."""
    print(f"Error in python_executor_node: {str(e)}")
    traceback.print_exc()
    
    # df_dict가 생성됐는지 확인하고 결과에 포함시키기
    try:
        # 마지막으로 생성된 df_dict를 결과에 포함
        
        if hasattr(builtins, 'df_dict'):
            result_df_dict = builtins.df_dict
            
            # 직접 변수에서 데이터프레임을 찾아 추가
            for var_name in dir(builtins):
                if var_name.startswith('__'):
                    continue
                var_value = getattr(builtins, var_name)
                if isinstance(var_value, pd.DataFrame):
                    print(f"Debug - Found '{var_name}' DataFrame in builtins during error recovery")
                    result_df_dict[var_name] = var_value
                    
            # df_dict 제거
            del builtins.df_dict
            
            return {
                "messages": state.messages,
                "question": state.question,
                "selected_tables": state.selected_tables,
                "generated_queries": state.generated_queries,
                "df_dict": result_df_dict,  # 결과에 df_dict 포함
                "python_code": None,  # 에러 시 Python 코드 없음
                "status": "error",
                "error": {"message": str(e)},
                "intermediate_steps": [],
            }
    except Exception as inner_e:
        print(f"Error in error handling: {str(inner_e)}")
    
    # 가능하면 builtins에서 df_dict 제거
    try:
        if hasattr(builtins, 'df_dict'):
            del builtins.df_dict
    except:
        pass
        
    return {
        "messages": state.messages,
        "question": state.question,
        "selected_tables": state.selected_tables,
        "generated_queries": state.generated_queries,
        "python_code": None,  # 에러 시 Python 코드 없음
        "status": "error",
        "error": {"message": str(e)},
        "intermediate_steps": [],
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