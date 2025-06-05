from typing import TypedDict, Dict, Any, List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field

class Evaluation(BaseModel):
    criterion: str = Field(..., description="Evaluation criterion text")
    score: float = Field(..., ge=0.0, le=1.0, description="A confidence score between 0.0 and 1.0")

class EvaluationResponse(BaseModel):
    evaluations: List[Evaluation] = Field(..., description="List of evaluation results by criteria")

class TopicEvaluation(BaseModel):
    topic: str = Field(description="Topics to evaluate (e.g., gaming, movies, travel, etc.)")
    prompt: str = Field(description="Evaluation Target prompt text")
    relevance_score: float = Field(
        description="A relevance score between 0.00 and 1.00. Up to two decimal places",
        ge=0.0,
        le=1.0,
        example=0.75,
    )
    reasoning: str = Field(description="A 2-3 sentence description of the relevance score")

class Thought(TypedDict):
    id: str
    text: str
    score: float
    reasoning: Optional[str] 
    depth: int
    parent_id: Optional[str]
    path_string: str

class CriterionResult(TypedDict):
    criterion: str
    criterion_en: str
    score: float
    evaluation: str

class CategoryEvaluation(TypedDict):
    criteria_results: List[CriterionResult] 
    average_score: float

EvaluationData = Dict[str, CategoryEvaluation]

class ReportState(TypedDict):
    dr_tavily_initial_context: Optional[str]
    dr_active_thoughts: List[Thought]
    dr_all_generated_thoughts: List[Thought]
    dr_selected_best_thought: Optional[Thought]
    dr_current_depth: int
    dr_max_depth: int
    dr_thoughts_per_expansion: int
    dr_top_k_to_keep: int
    dr_generated_for_evaluation: List[Thought]
    detailed_markdown_report: Optional[str]

class MainState(TypedDict):
    user_id: str
    token: str
    initial_prompt: str
    translated_prompt: Optional[str]
    optimized_prompt: Optional[str]
    is_completed: bool
    evaluation_data: Optional[EvaluationData]
    final_evaluation_data: Optional[Dict[str, Any]]
    improvement_suggestions: Optional[Dict[str, str]]
    human_feedback_ai: Optional[str]
    human_feedback: Optional[str]
    model: str
    topic: str
    error_message: Optional[str]
    generate_detailed_report: bool
    needs_web_search: bool
    web_search_data: Optional[List[Dict[str, Any]]]
    has_file_upload: bool
    uploaded_files: Optional[List[Dict[str, Any]]]
    report_data: Optional[ReportState]
    
criteria: Dict[str, List[tuple[str, str]]] = {
    "명확성": [
        ("문장이 명확한가?", "is clear and unambiguous"),
        ("모호성이 없는가?", "has no ambiguity"),
        ("핵심 정보가 잘 드러나는가?", "clearly presents key information"),
        ("질문 의도가 잘 표현되었는가?", "clearly expresses the intent of the question"),
        ("불필요한 단어가 없는가?", "contains no unnecessary words")
    ],
    "간결성": [
        ("짧고 간결하게 표현되었는가?", "is short and concise"),
        ("중복된 내용이 없는가?", "has no redundant content"),
        ("본질적인 정보만 포함되어 있는가?", "contains only essential information"),
        ("긴 문장이 너무 많지 않은가?", "does not have too many long sentences"),
        ("복잡한 구조 없이 간결한가?", "is concise without complex structures")
    ],
    "목표 적합성": [
        ("질문자의 목적과 일치하는가?", "aligns with the questioner's purpose"),
        ("원하는 정보만을 포함하고 있는가?", "contains only the desired information"),
        ("질문 대상이 명확한가?", "has a clear subject of inquiry"),
        ("질문 범위가 적절한가?", "has an appropriate scope of questioning"),
        ("원하는 스타일에 맞게 구성되어 있는가?", "is structured according to the desired style")
    ],
    "실행 가능성": [
        ("LLM이 쉽게 답변할 수 있는가?", "can be easily answered by an LLM"),
        ("논리적으로 말이 되는가?", "makes logical sense"),
        ("필요한 문맥이 포함되어 있는가?", "includes necessary context"),
        ("과도하게 난해하지 않은가?", "is not excessively complex"),
        ("LLM이 이해하기 쉬운가?", "is easy for an LLM to understand")
    ],
    "출력 품질 기대치": [
        ("원하는 형태로 출력될 수 있는가?", "can be output in the desired format"),
        ("다양한 사례를 요구하는가?", "requests various examples"),
        ("예시가 포함되어 있는가?", "includes examples"),
        ("특정 형식을 요구하는가?", "requests a specific format"),
        ("과도한 창의성이 필요한 질문이 아닌가?", "does not require excessive creativity")
    ],
    "사용자 만족도": [
        ("이전 대화 패턴과 일관적인가?", "is consistent with previous conversation patterns"),
        ("사용자의 스타일과 맞는가?", "matches the user's style"),
        ("개인화가 반영되었는가?", "reflects personalization"),
        ("피드백을 통해 개선되었는가?", "has been improved through feedback"),
        ("원하는 답변을 빠르게 얻을 수 있는가?", "can quickly get the desired answer")
    ]
}
