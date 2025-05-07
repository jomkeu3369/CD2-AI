from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

from rl_system import criterion_rl_data, save_weights, load_weights
from bart_evaluator import evaluate_prompt_with_rl

app = FastAPI(title="RL 테스트 서버")

class CriterionFeedbackRequest(BaseModel):
    criterion_key: str
    feedback_type: str

class PromptEvaluationRequest(BaseModel):
    prompt: str

@app.on_event("startup")
async def startup_event():
    load_weights()
    print("테스트 서버 시작: RL 가중치 로드됨.")

@app.post("/feedback_criterion/", summary="항목별 피드백 제출")
async def handle_criterion_feedback(request: CriterionFeedbackRequest):
    criterion_key = request.criterion_key
    feedback = request.feedback_type

    if criterion_key not in criterion_rl_data:
        raise HTTPException(status_code=404, detail=f"기준 항목 '{criterion_key}'을 찾을 수 없습니다.")

    current_data = criterion_rl_data[criterion_key]
    action_taken = ""

    if feedback == "recommend":
        current_data["recommend_score"] = current_data.get("recommend_score", 0) + 1
        current_data["weight"] = min(current_data.get("weight", 1.0) + 0.05, 2.0)
        action_taken = f"추천됨. 추천 점수: {current_data['recommend_score']}, 새 가중치: {current_data['weight']:.2f}"
        print(f"피드백 수신: [{criterion_key}] - 추천")
    elif feedback == "dont_recommend":
        current_data["weight"] = max(current_data.get("weight", 1.0) - 0.1, 0.1)
        action_taken = f"비추천됨. 새 가중치: {current_data['weight']:.2f}"
        print(f"피드백 수신: [{criterion_key}] - 비추천")
    else:
        raise HTTPException(status_code=400, detail="잘못된 피드백 유형입니다. 'recommend' 또는 'dont_recommend'를 사용하세요.")

    criterion_rl_data[criterion_key] = current_data
    save_weights()

    return {"message": f"피드백 처리 완료: '{criterion_key}'. {action_taken}"}

@app.post("/get_evaluation/", summary="프롬프트 평가 (RL 가중치 적용)")
async def get_prompt_evaluation(request: PromptEvaluationRequest) -> Dict[str, Any]:
    evaluation_results = evaluate_prompt_with_rl(request.prompt)
    return {"prompt": request.prompt, "evaluation": evaluation_results}

if __name__ == "__main__":
    print("테스트용 FastAPI 서버를 시작합니다. 주소: http://127.0.0.1:8008")
    uvicorn.run(app, host="127.0.0.1", port=8008)