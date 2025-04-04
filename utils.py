from typing import List
import pandas as pd
from datetime import datetime
import json

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
    import json
    import re

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
        # 4. 줄 단위로 정리 시도
        lines = json_part.split('\n')
        clean_lines = []

        for line in lines:
            # 주석이나 비정상적인 줄 제거
            if '//' in line:
                line = line[:line.index('//')]
            clean_lines.append(line.strip())

        clean_json = ' '.join(clean_lines)

        try:
            # 5. 다시 파싱 시도
            return json.loads(clean_json)

        except:
            # 6. 최후의 수단: 정규식으로 키-값 쌍 추출
            result = {}
            pattern = r'"([^"]*)"\s*:\s*"([^"]*)"'
            matches = re.findall(pattern, json_part)

            for key, value in matches:
                result[key] = value

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