from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class OllamaPromptRequest(BaseModel):
    prompt: str
    model: str = "gemma3:12b"
    # stream: bool = False

class OllamaGenerateResponse(BaseModel):
    generated_text: str | None = None
    error: str | None = None
    detail: Any | None = None


class BartClassifyRequest(BaseModel):
    sequence: str = Field(..., example="Apple just announced the new iPhone 15.")
    candidate_labels: List[str] = Field(..., example=["technology", "politics", "finance"])
    multi_label: bool = False

class BartClassifyResponse(BaseModel):
    sequence: str
    labels: List[str]
    scores: List[float]
    error: str | None = None
    detail: Any | None = None


class OptimizeRequest(BaseModel):
    prompt: str = Field(..., description="최적화할 원본 한국어 프롬프트", example="windows10 환경 변수 설정 방법 알려줘")
    thema: str

class OptimizeResponse(BaseModel):
    original_prompt: str                                    # 최초 프롬프트                  
    enhanced_prompt: Optional[str] = None                   # 최적화된 프롬프트
    english_prompt: Optional[str] = None                    # original_prompt를 DeepL을 이용해 영어로 번역된 프롬프트
    error_message: Optional[str] = None                     # 에러 발생 시 에러 메시지
    weak_categories: Optional[List[str]] = None             # BART 평가 데이터에서 보완할 항목 (평가 결과 0.5 이하) 
    suggestions_count: Optional[int] = None                 # 보완될 항목의 수
    evaluation_summary: Optional[Dict[str, float]] = None   # BART 평가 데이터
