from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Annotated
import pandas as pd

from serializer_utils import sanitize_intermediate_steps

class DBState(BaseModel):
    # 대화 히스토리
    messages: Annotated[List[Dict[str, Any]], "messages"] = Field(default_factory=list)
    question: Optional[str] = None
    
    # 각 도구의 출력
    selected_tables: Dict[str, str] = Field(default_factory=dict)
    generated_queries: Dict[str, str] = Field(default_factory=dict)
    df_dict: Dict[str, Any] = Field(default_factory=dict)
    python_code: Optional[str] = None
    final_result: Optional[str] = None
    result_df: Optional[Any] = None
    
    # 상태 관리
    status: str = "running"
    error: Optional[Dict[str, Any]] = None
    next: Optional[str] = None
    completed: Optional[bool] = False
    
    # 코드 피드백 정보
    code_feedback: Optional[Dict[str, str]] = None
    
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