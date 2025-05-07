import httpx
import asyncio
import random
from rl_criteria_config import criteria as schema_criteria

BASE_URL = "http://127.0.0.1:8008"

def get_random_criterion_key():
    category_name = random.choice(list(schema_criteria.keys()))
    criterion_details = random.choice(schema_criteria[category_name])
    return f"{category_name}::{criterion_details[0]}"

async def run_test_simulation():
    async with httpx.AsyncClient() as client:
        sample_prompt = "오늘 날씨 어때?"
        print(f"--- 초기 평가 수행: '{sample_prompt}' ---")
        try:
            response = await client.post(f"{BASE_URL}/get_evaluation/", json={"prompt": sample_prompt})
            response.raise_for_status()
            initial_eval = response.json()
            print("초기 평가 결과:")
            for category, data in initial_eval.get("evaluation", {}).items():
                print(f"  카테고리 [{category}]: 평균 점수 {data.get('average_score', 'N/A'):.4f}")
                for crit_res in data.get('criteria_results', []):
                    print(f"    - 기준: {crit_res['criterion']}, 원점수: {crit_res['raw_bart_score']:.2f}, RL가중치: {crit_res['rl_weight']:.2f}, 추천점수: {crit_res['recommend_score']}, 최종점수: {crit_res['final_score']:.2f}")

        except Exception as e:
            print(f"초기 평가 중 요청 오류: {e}")
        
        print("\n--- 무작위 피드백 제출 시뮬레이션 ---")
        for i in range(3):
            criterion_to_feedback = get_random_criterion_key()
            feedback_type = random.choice(["recommend", "dont_recommend"])
            print(f"{i+1}번째 피드백: 항목 '{criterion_to_feedback}', 유형 '{feedback_type}'")
            try:
                feedback_response = await client.post(
                    f"{BASE_URL}/feedback_criterion/",
                    json={"criterion_key": criterion_to_feedback, "feedback_type": feedback_type}
                )
                feedback_response.raise_for_status()
                print(f"피드백 결과: {feedback_response.json().get('message')}")
            except Exception as e:
                 print(f"피드백 제출 중 요청 오류 ({criterion_to_feedback}): {e}")

            await asyncio.sleep(0.1)

        print(f"\n--- 피드백 후 재평가 수행: '{sample_prompt}' ---")
        try:
            response_after_feedback = await client.post(f"{BASE_URL}/get_evaluation/", json={"prompt": sample_prompt})
            response_after_feedback.raise_for_status()
            eval_after_feedback = response_after_feedback.json()
            print("피드백 후 평가 결과:")
            for category, data in eval_after_feedback.get("evaluation", {}).items():
                print(f"  카테고리 [{category}]: 평균 점수 {data.get('average_score', 'N/A'):.4f}")
                for crit_res in data.get('criteria_results', []):
                    print(f"    - 기준: {crit_res['criterion']}, 원점수: {crit_res['raw_bart_score']:.2f}, RL가중치: {crit_res['rl_weight']:.2f}, 추천점수: {crit_res['recommend_score']}, 최종점수: {crit_res['final_score']:.2f}")
        except Exception as e:
            print(f"재평가 중 요청 오류: {e}")

if __name__ == "__main__":
    asyncio.run(run_test_simulation())