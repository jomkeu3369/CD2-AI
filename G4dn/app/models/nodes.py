from app.models.schema import MainState, criteria

from datetime import datetime
import os
import httpx
import json
import deepl
import numpy as np
import asyncio

from langchain.prompts import ChatPromptTemplate

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
client = httpx.AsyncClient()

async def trans_en(text:str) -> str:
    translator = deepl.Translator(DEEPL_API_KEY)
    result = translator.translate_text(text, target_lang="en-us")
    return str(result)

async def evaluate_text(text:str, criteria_dict:dict) -> dict:
    from app.models.classifiers import classifier

    english_text = text
    results = {}

    for category, criteria_items in criteria_dict.items():
        category_results = []

        for criterion_kr, criterion_en in criteria_items:
            result = classifier(
                english_text,
                candidate_labels=[criterion_en],
                hypothesis_template = "This text {}."
            )
            score = result['scores'][0]
            category_results.append({
                'criterion': criterion_kr,
                'criterion_en': criterion_en,
                'score': score,
                'evaluation': "충족함" if score > 0.5 else "충족하지 않음"
            })

        avg_score = np.mean([item['score'] for item in category_results])

        results[category] = {
            'criteria_results': category_results,
            'average_score': avg_score
        }

    return results

async def analyze_prompt(state: MainState) -> MainState:
    from app.models.classifiers import classifier
    user_prompt = state["original_prompt"]

    if classifier is None:
        state["execution_log"] = state["execution_log"] + [{
             "node": "analyze_prompt", 
             "error": "BART classifier not loaded.",
             "timestamp": datetime.now().isoformat()
        }]
        state["exit"] = True
        return state

    else:
        try:
            en_str = await trans_en(user_prompt)
            scores_dict = await evaluate_text(en_str, criteria)

        except Exception as e:
            state["execution_log"] = state["execution_log"] + [{
                "node": "analyze_prompt", 
                "error": f"Exception during classification: {e}",
                "timestamp": datetime.now().isoformat()
            }]
            state["exit"] = True
            return state

    state["evaluation_data"] = scores_dict
    state["execution_log"] = state["execution_log"] + [{
         "node": "analyze_prompt", 
         "status": "completed",
         "timestamp": datetime.now().isoformat()
    }]
    return state

async def analyze_evaluation_data(state: MainState) -> MainState:
    if "execution_log" not in state:
        state["execution_log"] = []

    weak_categories = []
    evaluation_data = state["evaluation_data"]

    if not evaluation_data:
         log_entry = {
            "node": "analyze_evaluation_data",
            "timestamp": datetime.now().isoformat(),
            "status": "skipped",
            "reason": "evaluation_data is empty",
            "weak_categories": weak_categories
         }
         state["weak_categories"] = weak_categories
         state["execution_log"] = state["execution_log"] + [log_entry]
         return state

    '''
        evaluation_data 에는 
    
    '''

    for category, data in evaluation_data.items():
         if not isinstance(data, dict):
             print(f"Invalid data format for category '{category}' in evaluation_data.")
             continue

         avg_score = data.get("average_score", 0)
         if avg_score < 0.6:
            weak_categories.append(category)

         for criterion in data.get("", []):
            # criterion이 dict가 아닌 경우를 대비
            if not isinstance(criterion, dict):
                print(f"Warning: Invalid criterion format in category '{category}'.")
                continue

            if criterion.get("score", 0) < 0.4:
                weak_item = f"{category}::{criterion.get('criterion', 'Unknown Criterion')}"
                if weak_item not in weak_categories:
                    weak_categories.append(weak_item)

    log_entry = {
        "node": "analyze_evaluation_data",
        "timestamp": datetime.now().isoformat(),
        "weak_categories": weak_categories
    }

    state["weak_categories"] = weak_categories
    state["execution_log"] = state.get("execution_log", []) + [log_entry]
    return state

async def _get_suggestion_for_category(category: str, original_prompt: str, evaluation_data: dict, http_client: httpx.AsyncClient, ollama_url: str) -> tuple[str, str]:
    suggestion_text = f"개선 제안 생성 실패 (카테고리: {category})"
    try:
        if "::" in category:
            main_cat, specific_criterion = category.split("::", 1)
        else:
            main_cat, specific_criterion = category, "전반적 개선"

        category_data = evaluation_data.get(main_cat, {})
        criteria_results = category_data["criteria_results"]

        prompt_template = ChatPromptTemplate.from_template(
            """
            Please improve the following prompt in terms of '{category}':

            [original prompt]
            {original_prompt}

            [Evaluation results for {category}]
            {evaluation_results}

            In particular, please focus on the '{specific_criterion}' items in particular, and provide suggestions for improvement in Korean. 
            Provide only the improved prompt suggestion.
            """
        )

        formatted_prompt = prompt_template.format(
            category=main_cat,
            original_prompt=original_prompt,
            evaluation_results=json.dumps(criteria_results, ensure_ascii=False, indent=2),
            specific_criterion=specific_criterion
        )

        response = await http_client.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "gemma3:12b",
                "prompt": formatted_prompt,
                "stream": False,
            },
            headers={'Content-Type': 'application/json'},
            timeout=120.0
        )

        response.raise_for_status() # HTTP 에러 발생 시 예외 발생
        ollama_result = response.json()
        generated_content = ollama_result.get("response", "").strip()

        if ollama_result.get("done"):
            suggestion_text = generated_content if generated_content else "모델이 응답을 생성하지 못했습니다."
        else:
            suggestion_text = f"Ollama가 생성을 완료하지 못했습니다: {ollama_result}"

    except httpx.RequestError as e:
        suggestion_text = f"Ollama API 호출 실패: {str(e)}"
        print(f"Error getting suggestion for {category}: {suggestion_text}")
    except httpx.HTTPStatusError as e:
         suggestion_text = f"Ollama API 서버 오류 (Status {e.response.status_code}): {e.response.text}"
         print(f"Error getting suggestion for {category}: {suggestion_text}")
    except Exception as e:
        suggestion_text = f"제안 생성 중 예외 발생: {str(e)}"
        print(f"Error getting suggestion for {category}: {suggestion_text}")

    return category, suggestion_text

async def generate_improvement_suggestions(state: MainState) -> MainState:

    weak_categories = state["weak_categories"]
    original_prompt = state.get("original_prompt", "원본 프롬프트 정보 없음")
    evaluation_data = state["evaluation_data"]

    tasks = []
    if weak_categories: # weak_categories가 있을 때만 작업 생성
        for category in weak_categories:
            tasks.append(
                _get_suggestion_for_category(
                    category=category,
                    original_prompt=original_prompt,
                    evaluation_data=evaluation_data,
                    http_client=client, # 전역 클라이언트 사용
                    ollama_url=OLLAMA_URL
                )
            )

    suggestions = {}
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 처리
        for result in results:
            if isinstance(result, Exception):
                print(f"An unexpected error occurred during suggestion generation: {result}")
            elif isinstance(result, tuple) and len(result) == 2:
                category, suggestion_text = result
                suggestions[category] = suggestion_text
            else:
                # 예상치 못한 결과 타입
                print(f"Unexpected result type from gather: {type(result)}")

    else:
        print("No weak categories found, skipping suggestion generation.")


    log_entry = {
        "node": "generate_improvement_suggestions",
        "timestamp": datetime.now().isoformat(),
        "suggestions_count": len(suggestions),
        "weak_categories_count": len(weak_categories),
        "model_used": "gemma3:12b"
    }

    current_log = state.get("execution_log", [])
    state["improvement_suggestions"] = suggestions
    state["execution_log"] = current_log + [log_entry]
    return state # 상태 객체 전체 반환

async def enhance_prompt(state: MainState) -> MainState:
    suggestions_context = "\n\n".join([
        f"[{category}에 대한 개선 제안]\n{suggestion}"
        for category, suggestion in state.get("improvement_suggestions", {}).items()
    ])

    evaluation_summary = {}
    evaluation_data = state["evaluation_data"]
    if evaluation_data:
         for category, data in evaluation_data.items():
            if not isinstance(data, dict): continue

            criteria_results = data.get("criteria_results", [])
            weak_criteria = [
                item.get("criterion", "Unknown") for item in criteria_results
                if isinstance(item, dict) and item.get("score", 0) < 0.5
            ]
            if "average_score" in data or weak_criteria:
                evaluation_summary[category] = {
                    "평균 점수": float(data.get("average_score", 0)),
                    "취약 항목": weak_criteria
                }

    prompt_template = ChatPromptTemplate.from_template(
        """
        [original prompt]
        {original_prompt}

        [Evaluation Summary]
        {evaluation_summary}

        [Suggestions for improvement]
        {suggestions_context}

        Based on the information above, please improve the original prompt. Pay attention to the following principles:

        1. Clarity: Remove ambiguity, emphasize key information. Ensure the request is specific.
        2. Conciseness: Remove redundancy, include only essential information. Be direct.
        3. Goal Fit: Align with the likely intent behind the original prompt.
        4. Feasibility: Phrase in a way that an LLM can easily understand and respond to effectively.
        5. Output Quality Control: If possible, specify the desired format, tone, or constraints for the output.

        Generate only the improved prompt, without any additional explanation or preamble. Output the result in Korean.
        """
    )

    formatted_prompt = prompt_template.format(
        original_prompt=state.get("original_prompt", ""),
        evaluation_summary=json.dumps(evaluation_summary, ensure_ascii=False, indent=2),
        suggestions_context=suggestions_context
    )

    enhanced_prompt = state.get("original_prompt", "") + "\n\n(최적화 실패: 기본값 반환)"
    error_detail = None # 에러 상세 정보 기록용

    try:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "gemma3:12b",
                "prompt": formatted_prompt,
                "stream": False,
            },
            headers={'Content-Type': 'application/json'},
            timeout=180.0
        )
        response.raise_for_status()

        ollama_result = response.json()
        generated_content = ollama_result.get("response", "").strip()

        if ollama_result.get("done") and generated_content:
            enhanced_prompt = generated_content
        else:
             error_detail = f"Ollama 응답 문제: {ollama_result}"
             print(f"프롬프트 최적화 실패: {error_detail}")
             enhanced_prompt = state.get("original_prompt", "") + f"\n\n(최적화 실패: {error_detail})"


    except httpx.RequestError as e:
        error_detail = f"API 요청 오류: {str(e)}"
        print(f"프롬프트 최적화 오류 (Request Error): {error_detail}")
        enhanced_prompt = state.get("original_prompt", "") + f"\n\n(최적화 실패: {error_detail})"
    except httpx.HTTPStatusError as e:
        error_detail = f"API 상태 오류 {e.response.status_code}: {e.response.text}"
        print(f"프롬프트 최적화 오류 (HTTP Status Error): {error_detail}")
        enhanced_prompt = state.get("original_prompt", "") + f"\n\n(최적화 실패: API 상태 오류 {e.response.status_code})"
    except Exception as e:
        error_detail = f"내부 오류: {str(e)}"
        print(f"프롬프트 최적화 오류 (Unknown Exception): {error_detail}")
        enhanced_prompt = state.get("original_prompt", "") + f"\n\n(최적화 실패: {error_detail})"

    # 상태 업데이트
    log_entry = {
        "node": "enhance_prompt",
        "timestamp": datetime.now().isoformat(),
        "original_length": len(state.get("original_prompt", "")),
        "enhanced_length": len(enhanced_prompt),
        "model_used": "gemma3:12b",
        "error_detail": error_detail # 에러 발생 시 상세 내용 로깅
    }

    current_log = state.get("execution_log", [])
    state["enhanced_prompt"] = enhanced_prompt
    state["execution_log"] = current_log + [log_entry]
    return state # 상태 객체 전체 반환
