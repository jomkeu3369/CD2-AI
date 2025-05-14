from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class OptimizeRequest(BaseModel):
    prompt: str = Field(..., description="최적화할 원본 한국어 프롬프트", example="windows10 환경 변수 설정 방법 알려줘")
    topic: str

class OptimizeResponse(BaseModel):               
    enhanced_prompt: str                 # 최적화된 프롬프트
