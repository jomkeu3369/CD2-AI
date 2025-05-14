from app.model.schema import MainState, criteria, EvaluationData

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI

from fastapi import WebSocket
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any

import os
import json
import deepl
import numpy as np
import asyncio

load_dotenv()
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
translator = deepl.Translator(auth_key=DEEPL_API_KEY)

'''
    -1. 주제 평가 노드

    0. 프롬프트 번역 노드
        * 프롬프트를 영어로 번역하는 노드

    1. 프롬프트 평가 노드
        * LLM as a judge 방식으로 프롬프트를 평가
        
    2. 보완 노드

    3. 융합 노드

    4. 프롬프트 평가 노드

    5. 라우터1

    6. 라우터2

'''

# ---------------------------
#  기능 함수
# ---------------------------

async def text_translation(text:str, end_lang:str="EN-US") -> str:
    """ 텍스트를 번역합니다. """
    try:
        
        result = translator.translate_text(text=text, target_lang=end_lang)
        return str(result).strip()
    except:
        return text

async def evaluate_text(text: str, criteria_list: List[str], model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    criteria_string = "\n".join(f'{idx}. "{criterion}"' for idx, criterion in enumerate(criteria_list, start=1))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert text evaluator. Your task is to evaluate a given text against a list of criteria.
For each criterion, determine if the text meets the criterion and provide a confidence score between 0.0 and 1.0 (where 1.0 means a perfect match and 0.0 means no match).
Your response MUST be a valid JSON object. The JSON object should have a single key "evaluations".
The value of "evaluations" should be a list of objects, where each object corresponds to one criterion from the input list (in the same order) and has three keys:
1. "criterion": The exact criterion string.
2. "score": A float value between 0.0 and 1.0 representing the confidence score.
Example for one criterion: "criterion": "example criterion text", "score": 0.81"""),
        
        ("user", """Please evaluate the following text:
--- TEXT START ---
{text}
--- TEXT END ---

Against these criteria:
--- CRITERIA START ---
{criteria_string}
--- CRITERIA END ---

Provide your evaluation as a JSON object as specified in the system prompt.""")
    ])
    
    model = ChatOpenAI(
        model=model_name,
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    chain = prompt | model | StrOutputParser()
    
    try:
        response = await chain.ainvoke({"text": text, "criteria_string": criteria_string})
        parsed_response = json.loads(response)
        return parsed_response
        
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}

async def enhance_prompt(existing_prompt: str, improvement_points: str, model_name: str = "gpt-4o-mini") -> dict:
    system_template = """
    You are an expert prompt enhancer. Your task is to improve the given prompt by addressing the specified improvement points.
    
    Given the existing prompt and the points that need improvement, generate an enhanced prompt that incorporates the improvements clearly and effectively.
    
    Provide only the enhanced prompt as a plain text response without any additional explanation.
    """

    user_template = """
    Existing prompt:
    {existing_prompt}

    Improvement points:
    {improvement_points}

    Please provide the enhanced prompt incorporating the improvement points.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])

    model = ChatOpenAI(
        model=model_name,
        temperature=0.7
    )

    chain = prompt | model | StrOutputParser()

    try:
        response = await chain.ainvoke({
            "existing_prompt": existing_prompt,
            "improvement_points": improvement_points
        })
        
        return {improvement_points: response}
        
    except Exception as e:
        return {"error": f"Error enhancing prompt: {str(e)}"}

async def generate_final_korean_prompt(original_english_prompt: str, enhancement_results: Dict[str, str], model_name: str) -> str:
    enhancements_text = ""
    for idx, (category, suggestion) in enumerate(enhancement_results.items(), 1):
        enhancements_text += f"\nImprovement {idx} - Category: {category}\n{suggestion}\n"

    system_template = """
    You are a prompt engineering expert. Your task is to integrate the original prompt with multiple improvement suggestions to create a final enhanced prompt.
    Carefully analyze the original prompt and all provided improvement suggestions.
    Then, create a comprehensive final prompt that incorporates all valuable aspects of the improvements while maintaining consistency and clarity.

    IMPORTANT: The final enhanced prompt MUST be in KOREAN. Do not provide any explanations or notes outside of the KOREAN prompt itself.
    """
    user_template = """
    Original Prompt (for context):
    {original_english_prompt}

    Improvement Suggestions (based on the original prompt):
    {enhancements_text} # --- 여기서 enhancements_text가 사용됩니다 ---

    Please write a final, enhanced prompt in KOREAN that effectively integrates all valuable improvements.
    Provide only the KOREAN prompt.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])

    model_instance = ChatOpenAI(
        model=model_name,
        temperature=0.5,
        streaming=True
    )
    chain = prompt_template | model_instance | StrOutputParser()

    accumulated_korean_response = ""
    async for korean_chunk in chain.astream({
        "original_english_prompt": original_english_prompt,
        "enhancements_text": enhancements_text
    }):
        if korean_chunk:
            accumulated_korean_response += korean_chunk
    return accumulated_korean_response

# ---------------------------
#  LangGraph 노드 함수
# ---------------------------

# 주제 평가 노드
async def topic_evaluation_node(status: MainState):
    topic = status['topic']
    initial_prompt = status["initial_prompt"]
    model_name = status["model"]

    system_template = """
        You are an expert evaluator who objectively assesses the relevance between prompts and topics. Your task is to determine how relevant a given prompt is to a specific topic (such as "gaming", "movies", "travel", "food", etc.).

        Evaluate the relevance on a continuous scale from 0.00 to 1.00, where:
        - 0.00 represents absolutely no relevance
        - 1.00 represents complete and direct relevance

        You can use any value between 0.00 and 1.00 with two decimal places (e.g., 0.37, 0.82, etc.) based on your assessment. Do not limit yourself to specific increments.

        Here are some reference points on the scale to guide your thinking:
        - Around 0.00: No connection at all between the prompt and topic
        - Around 0.20-0.30: Minimal relevance, only tangential connections
        - Around 0.40-0.60: Moderate relevance, some clear connections but not the main focus
        - Around 0.70-0.80: Strong relevance, topic is a significant focus of the prompt
        - Around 0.90-1.00: Very high to complete relevance, prompt is centered on the topic

        Your evaluation should be based on objective criteria and avoid subjective preferences or personal opinions. The score should reflect how centrally the topic appears in the prompt.

        Provide your response in the following JSON format:
        "topic": "the evaluation topic",
        "prompt": "the evaluated prompt",
        "relevance_score": X.XX,
        "reasoning": "detailed explanation of your assessment (2-3 sentences)"
    """

    user_template = """
        Please evaluate the relevance between the following prompt and topic on a scale from 0.00 to 1.00:

        Topic: "{topic}"
        Prompt: "{prompt}"

        Provide the relevance score and your reasoning in JSON format.
    """

    model = ChatOpenAI(model=model_name, max_tokens=4096, temperature=0)

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(user_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_message_prompt,
            human_message_prompt
        ]
    )
    chain = chat_prompt | model | StrOutputParser()
    generation = await chain.ainvoke(input={"topic": topic, "prompt": initial_prompt})
    
    parsed_output:dict = json.loads(generation)
    if parsed_output.get("relevance_score", 1.0) < 0.6:
        status["is_completed"] = True

    return status

# 프롬프트 번역 노드
async def translation_prompt_node(status: MainState):
    initial_prompt = status["initial_prompt"]

    en_text = await text_translation(initial_prompt)
    
    status["translated_prompt"] = en_text
    return status

# 프롬프트 평가 노드
async def evaluate_prompt_node(status: MainState):

    en_text = status["translated_prompt"]
    model = status["model"]

    en_all_criteria = []
    flat_criteria_list = [] 

    # 카테고리 분류
    for category, criteria_pairs in criteria.items():
        for kr_criterion, en_criterion in criteria_pairs:
            en_all_criteria.append(en_criterion)
            flat_criteria_list.append((category, kr_criterion, en_criterion))
    
    results = {category_name: {'criteria_results': [], 'average_score': 0.0} for category_name in criteria.keys()}
    
    try:
        api_result = await evaluate_text(en_text, en_all_criteria, model)
    except:
        status["is_completed"] = True
        return status
    
    # 에러가 발생한 경우
    if isinstance(api_result, dict) and 'error' in api_result:
        status["is_completed"] = True
        return status
    
    # 포멧 형식의 개수가 맞지 않는 경우
    elif len(api_result['evaluations']) != len(flat_criteria_list):
        status["is_completed"] = True
        return status
    
    # 스코어 기록
    for i, (category, init_criterion, en_criterion) in enumerate(flat_criteria_list):
        current_evaluation_text = ""

        eval_item:dict = api_result['evaluations'][i]
        current_score = float(eval_item.get("score", 0.0))
        current_evaluation_text = "충족함" if current_score > 0.5 else "충족하지 않음"
        
        results[category]['criteria_results'].append({
            'criterion': init_criterion,
            'criterion_en': en_criterion,
            'score': current_score,
            'evaluation': current_evaluation_text
        })

    # 평균값 계산
    for category_name, category_data in results.items():
        valid_scores = [item['score'] for item in category_data['criteria_results']]
        average_category_score = np.mean(valid_scores) if valid_scores else 0.0
        results[category_name]['average_score'] = average_category_score
            
    status["evaluation_data"] = results
    return status 

# 프롬프트 보완 노드
async def improvement_prompt_node(status: MainState):
    translated_prompt = status["translated_prompt"]
    evaluation_data: EvaluationData = status["evaluation_data"]
    model = status["model"]
    
    vulnerable_categories_names: List[str] = []
    IMPROVEMENT_THRESHOLD = 0.6 

    # 보완이 필요한 항목을 기록 (IMPROVEMENT_THRESHOLD 값 미만.)
    for main_category_name, category_data in evaluation_data.items():
        average_score = category_data['average_score']
        
        if average_score < IMPROVEMENT_THRESHOLD:
            for sub_category in category_data["criteria_results"]:
                
                if sub_category["score"] < IMPROVEMENT_THRESHOLD:
                    vulnerable_categories_names.append(sub_category["criterion_en"])
    
    # 병렬 테스트로 프롬프트 보완
    suggestions = {}
    if vulnerable_categories_names:
        tasks = []

        for vulnerable_category in vulnerable_categories_names:
            tasks.append(
                enhance_prompt(translated_prompt, vulnerable_category, model)
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            print(f"출력값 : {result}")
            suggestions.update(result)

    status["improvement_suggestions"] = suggestions
    return status

# Human in the loop 노드

# 웹 검색 노드 (MCP)

# 파일 로딩 노드

# 프롬프트 융합 노드
async def enhance_prompt_node(status: MainState):

    english_context_prompt = status["translated_prompt"]
    improvement_suggestions = status["improvement_suggestions"]
    model = status["model"]

    korean_optimized_prompt = await generate_final_korean_prompt(
        english_context_prompt,
        improvement_suggestions,
        model
    )
    status["optimized_prompt"] = korean_optimized_prompt
    status["is_completed"] = False 
    return status

# 프롬프트 재평가 노드
async def evaluate_enhance_prompt_node(status: MainState, config: dict):
    websocket = config.get("configurable", {}).get("websocket")
    final_korean_prompt = status["optimized_prompt"] # enhance_prompt_node에서 생성된 한국어 프롬프트
    model_name = status["model"]

    if not final_korean_prompt:
        print("최종 한국어 프롬프트가 없어 평가를 진행할 수 없습니다.")
        # 이 경우, 평가 데이터를 비우거나 특정 상태로 표시
        status["final_evaluation_data"] = {} # 새로운 상태 키 사용 (예시)
        return status

    # 한국어 프롬프트 평가를 위한 기준 (기존 criteria 사용 또는 한국어용 별도 기준)
    # 여기서는 기존 criteria의 한국어 기준명(첫 번째 요소)을 사용한다고 가정
    # 또는 LLM-as-a-judge 프롬프트를 한국어 평가에 맞게 수정

    # evaluate_text 함수를 한국어 평가에 맞게 사용하거나,
    # 한국어 평가를 위한 별도의 LLM 호출 로직 구성 필요.
    # 다음은 기존 evaluate_text를 사용한다고 가정하고, 평가 기준 전달 방식을 보여주는 예시입니다.
    # 실제로는 evaluate_text 내부의 LLM 프롬프트가 한국어 텍스트 평가에 적합해야 합니다.

    # 한국어 기준 목록 생성 (예시)
    korean_criteria_list = []
    for category, criteria_pairs in criteria.items():
        for kr_criterion, _ in criteria_pairs:
            korean_criteria_list.append(kr_criterion)

    # `evaluate_text` 함수가 한국어 텍스트와 한국어 기준 목록을 잘 처리할 수 있도록
    # 내부 프롬프트를 확인하거나, 한국어 평가 전용 함수를 만들어야 할 수 있습니다.
    # 여기서는 개념적으로 `evaluate_text`를 호출한다고 가정합니다.
    # LLM이 한국어 평가 기준과 한국어 프롬프트를 이해하고 JSON 형식으로 점수를 매기도록 해야 합니다.
    # 해당 LLM 프롬프트는 "is clear and unambiguous" 대신 "명확하고 모호하지 않은가?" 와 같은
    # 한국어 기준으로 평가하도록 지시해야 합니다.
    try:
        # 만약 evaluate_text의 LLM 프롬프트가 영어 기준만 가정한다면,
        # 이 부분은 한국어 평가를 위한 새로운 LLM 호출로 대체되어야 합니다.
        # 예를 들어, 각 기준에 대해 "이 프롬프트는 [한국어 기준]을 만족하는가? (점수 0.0-1.0)" 와 같이 LLM에 직접 질문.
        # 여기서는 'evaluate_text'가 한국어 프롬프트와 기준(영문 기준 설명이지만)으로 평가를 시도한다고 가정.
        # 실제로는 LLM이 한국어 평가를 잘 하도록 프롬프트 수정이 필수적입니다.
        evaluation_results_raw = await evaluate_text(
            text=final_korean_prompt, 
            criteria_list=[en_crit for _, pair_list in criteria.items() for _, en_crit in pair_list], # 임시로 영어 기준 사용
            model_name=model_name
        )
        # 결과 처리 로직은 evaluate_prompt_node와 유사하게 적용
        # ... (결과 파싱 및 status["final_evaluation_data"]에 저장) ...
        # 이 부분은 evaluate_prompt_node의 결과 처리 로직을 참고하여 채워야 합니다.
        # 예시:
        parsed_results = {} # evaluate_text 결과 파싱한 결과라 가정
        # status["final_evaluation_data"] = parsed_results 

        # 임시로 평균 점수 계산 (실제로는 evaluate_prompt_node 처럼 상세히)
        # 다음은 매우 단순화된 예시입니다. 실제로는 evaluate_prompt_node의 점수 계산 로직을 따라야 합니다.
        avg_score = 0.0
        num_criteria = 0
        if "evaluations" in evaluation_results_raw:
            scores = [item.get("score", 0.0) for item in evaluation_results_raw["evaluations"]]
            if scores:
                avg_score = sum(scores) / len(scores)
            num_criteria = len(scores)

        print(f"최종 한국어 프롬프트 평가 점수 (평균): {avg_score} ({num_criteria}개 기준)")
        status["final_evaluation_data"] = {"average_score": avg_score, "details": evaluation_results_raw} # 상세 결과 저장

        # (선택) 웹소켓으로 평가 진행 중 또는 결과 알림
        if websocket:
            await websocket.send_json({
                "type": "status_update", # 또는 "intermediate_result"
                "text": f"최종 프롬프트 평가 완료. 평균 점수: {avg_score:.2f}"
            })

    except Exception as e:
        print(f"최종 한국어 프롬프트 평가 중 오류: {e}")
        status["final_evaluation_data"] = {"error": str(e)}

    return status

# 답변 노드 (프롬프트 최적화)
async def stream_final_answer_node(status: MainState, config: dict):
    websocket = config.get("configurable", {}).get("websocket")
    final_answer_to_stream = status.get("optimized_prompt")

    if websocket and final_answer_to_stream:
        chunk_size = 50
        for i in range(0, len(final_answer_to_stream), chunk_size):
            chunk = final_answer_to_stream[i:i+chunk_size]
            await websocket.send_json({
                "type": "result",
                "text": chunk
            })
            await asyncio.sleep(0.05)

        status["is_completed"] = True
    elif not final_answer_to_stream:
        print("스트리밍할 최종 답변이 없습니다.")
        if websocket:
             await websocket.send_json({"type": "error", "text": "최종 답변을 생성하지 못했습니다."})
        status["is_completed"] = True
    else:
         print("웹소켓이 없어 최종 답변을 스트리밍 할 수 없습니다.")
         status["is_completed"] = True
    return status

# 재시도 시 상태 초기화/수정 노드
async def prepare_for_retry_node(status: MainState):
    current_retry_count = status.get("retry_count", 0)
    status["retry_count"] = current_retry_count + 1
    status["final_evaluation_data"] = None 
    return status

# TOT 메인 노드

# TOT 서브 노드

# 최종 피드백 생성 노드