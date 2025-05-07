import random
from rl_criteria_config import criteria as schema_criteria
from rl_system import criterion_rl_data

def simulate_bart_classification(sequence, candidate_label):
    return random.uniform(0.3, 0.9)

def evaluate_prompt_with_rl(prompt_text: str):
    results = {}
    print(f"\n'({prompt_text})' 프롬프트에 대한 RL 가중치 적용 평가 시작:")

    for category, criteria_items in schema_criteria.items():
        category_results = []
        weighted_scores_sum = 0
        total_weights_for_category = 0

        for criterion_kr, criterion_en in criteria_items:
            criterion_key = f"{category}::{criterion_kr}"
            
            raw_bart_score = simulate_bart_classification(prompt_text, criterion_en)
            
            rl_data = criterion_rl_data.get(criterion_key, {"weight": 1.0, "recommend_score": 0})
            current_weight = rl_data.get("weight", 1.0)
            recommend_score = rl_data.get("recommend_score", 0)
            
            weighted_score = raw_bart_score * current_weight
            
            category_results.append({
                'criterion': criterion_kr,
                'raw_bart_score': round(raw_bart_score, 4),
                'rl_weight': round(current_weight, 4),
                'recommend_score': recommend_score,
                'final_score': round(weighted_score, 4),
            })
            
            weighted_scores_sum += weighted_score
            total_weights_for_category += current_weight

        if total_weights_for_category > 0:
            avg_score_for_category = weighted_scores_sum / total_weights_for_category
        elif category_results:
            raw_scores = [item['raw_bart_score'] for item in category_results]
            avg_score_for_category = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
        else:
            avg_score_for_category = 0.0
            
        results[category] = {
            'criteria_results': category_results,
            'average_score': round(avg_score_for_category, 4)
        }
        print(f"카테고리 [{category}] 평균 점수 (가중치 적용): {results[category]['average_score']:.4f}")
        # for res_item in category_results:
        #     print(f"  - 기준: {res_item['criterion']}, 원점수: {res_item['raw_bart_score']:.2f}, 가중치: {res_item['rl_weight']:.2f}, 추천점수: {res_item['recommend_score']}, 최종점수: {res_item['final_score']:.2f}")

    return results