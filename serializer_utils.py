"""
직렬화 관련 유틸리티 함수를 제공하는 모듈입니다.
특히 DataFrame과 같은 복잡한 객체들을 JSON으로 직렬화할 때 발생하는 문제를 해결합니다.
"""
import pandas as pd
import numpy as np
import inspect
from typing import Any, Dict, List, Optional, Union
import traceback


def is_serializable(obj: Any) -> bool:
    """
    객체가 직렬화 가능한지 확인합니다.
    """
    try:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return True
        elif isinstance(obj, (list, tuple)):
            return all(is_serializable(item) for item in obj)
        elif isinstance(obj, dict):
            return all(isinstance(k, str) and is_serializable(v) for k, v in obj.items())
        else:
            # 직렬화 불가능한 객체
            return False
    except:
        return False


def safe_serialize_value(value: Any) -> Any:
    """
    객체를 안전하게 직렬화 가능한 형태로 변환합니다.
    """
    if value is None:
        return None
    
    # DataFrame 처리
    if isinstance(value, pd.DataFrame):
        return f"<DataFrame: {value.shape[0]}행 x {value.shape[1]}열>"
    
    # Series 처리
    if isinstance(value, pd.Series):
        return f"<Series: {len(value)}개 항목>"
    
    # Numpy array 처리
    if isinstance(value, np.ndarray):
        return f"<ndarray: shape={value.shape}, dtype={value.dtype}>"
    
    # 콜러블 객체 처리
    if callable(value):
        return f"<function: {value.__name__}>" if hasattr(value, "__name__") else "<function>"
    
    # 기본 타입은 그대로 반환
    if isinstance(value, (str, int, float, bool)):
        return value
    
    # 리스트/튜플 처리 - 재귀적으로 각 항목 처리
    if isinstance(value, (list, tuple)):
        return [safe_serialize_value(item) for item in value]
    
    # 딕셔너리 처리 - 재귀적으로 각 항목 처리
    if isinstance(value, dict):
        return {k: safe_serialize_value(v) for k, v in value.items()}
    
    # 객체 처리 - 문자열로 변환
    try:
        return str(value)
    except:
        return f"<객체: {type(value).__name__}>"


def deep_sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    딕셔너리를 재귀적으로 탐색하여 모든 DataFrame 등 직렬화 불가능한 객체를 제거합니다.
    """
    if not isinstance(data, dict):
        return safe_serialize_value(data)
    
    result = {}
    
    for key, value in data.items():
        # 특수 필드 직접 제외
        if key in ['df_dict', 'result_df', 'conf', 'dbm', 'db'] and not is_serializable(value):
            continue
            
        # 중첩된 딕셔너리 처리
        if isinstance(value, dict):
            result[key] = deep_sanitize_dict(value)
        # 리스트/튜플 처리
        elif isinstance(value, (list, tuple)):
            result[key] = [deep_sanitize_dict(item) if isinstance(item, dict) else safe_serialize_value(item) for item in value]
        # 기타 객체 처리
        else:
            result[key] = safe_serialize_value(value)
            
    return result


def sanitize_intermediate_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    중간 단계 기록에서 직렬화 불가능한 객체들을 안전하게 처리합니다.
    """
    sanitized_steps = []
    
    for step in steps:
        sanitized_step = {}
        
        # Action 처리
        if 'action' in step:
            action = step['action']
            if isinstance(action, dict):
                sanitized_step['action'] = deep_sanitize_dict(action)
            else:
                sanitized_step['action'] = str(action)
                
        # Observation 처리
        if 'observation' in step:
            observation = step['observation']
            if isinstance(observation, pd.DataFrame):
                sanitized_step['observation'] = f"<DataFrame: {observation.shape[0]}행 x {observation.shape[1]}열>"
            elif isinstance(observation, dict):
                sanitized_step['observation'] = deep_sanitize_dict(observation)
            else:
                sanitized_step['observation'] = str(observation)
                
        sanitized_steps.append(sanitized_step)
        
    return sanitized_steps


def sanitize_state_for_serialization(state: Any) -> Dict[str, Any]:
    """
    상태 객체를 직렬화 가능한 형태로 변환합니다.
    이 함수는 pydantic의 model_dump를 대체하여 사용합니다.
    """
    try:
        # Pydantic 모델의 경우
        if hasattr(state, 'model_dump'):
            # 기본 model_dump 실행 (재귀 호출 방지를 위해 상위 클래스의 메서드 사용)
            if hasattr(state.__class__, '__mro__'):
                for base_class in state.__class__.__mro__[1:]:  # 부모 클래스부터 시작
                    if hasattr(base_class, 'model_dump'):
                        model_dump_method = getattr(base_class, 'model_dump')
                        data = model_dump_method(state)
                        break
                else:
                    # 안전한 대체 방법 - __dict__ 사용
                    data = vars(state) if hasattr(state, '__dict__') else {}
            else:
                # 안전한 대체 방법
                data = vars(state) if hasattr(state, '__dict__') else {}
        # 딕셔너리의 경우
        elif isinstance(state, dict):
            data = state.copy()
        # 다른 객체의 경우 (예외 상황)
        else:
            data = vars(state) if hasattr(state, '__dict__') else {'error': 'Unsupported state object'}
            
        # DataFrame 관련 필드 제거
        for field in ['df_dict', 'result_df', 'db', 'dbm', 'conf']:
            if field in data:
                data.pop(field)
                
        # 중간 단계 처리
        if 'intermediate_steps' in data and data['intermediate_steps']:
            data['intermediate_steps'] = sanitize_intermediate_steps(data['intermediate_steps'])
            
        # 재귀적으로 모든 필드 안전 처리
        sanitized_data = deep_sanitize_dict(data)
        
        return sanitized_data
        
    except Exception as e:
        # 오류 발생 시 기본 정보만 반환
        print(f"Error sanitizing state: {str(e)}")
        # 재귀 오류 방지를 위해 traceback 출력 제거
        # traceback.print_exc()
        return {
            'status': 'error',
            'error': {'message': f"Serialization error: {type(e).__name__}"}
        } 