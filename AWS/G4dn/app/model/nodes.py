import asyncio
import uuid
import re
import numpy as np
import os
import logging
from typing import List, Optional, Dict, Any

from starlette.websockets import WebSocketState, WebSocketDisconnect
from tavily import TavilyClient

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

from app.api.crud import session_manager, session_weights
from app.model.schema import MainState, ReportState, criteria, Thought, TopicEvaluation
from app.model.crud import text_translation, evaluate_text, enhance_prompt, generate_final_prompt, human_in_the_loop_prompt
from app.log import setup_logging

logger: logging.Logger = setup_logging()
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# cot 출력
async def send_cot_message_to_websocket(config: dict, message: str):
    websocket = config.get("configurable", {}).get("websocket")
    if websocket and websocket.client_state == WebSocketState.CONNECTED:
        try:
            await websocket.send_json({"type": "cot", "text": message})
        except Exception as e:
            logger.error(f"Failed to send COT message: {e}", exc_info=True)

# 주제 판별 노드
async def topic_evaluation_node(status: MainState, config: dict) -> Dict[str, Any]:
    topic = status['topic']
    initial_prompt = status["initial_prompt"]
    model_name = status["model"]

    await send_cot_message_to_websocket(config, f"Starting relevance analysis for topic '{topic}' and prompt...")

    parser = JsonOutputParser(pydantic_object=TopicEvaluation)
    system_template = """
        You are an expert evaluator who objectively assesses the relevance between prompts and topics. 
        Your task is to determine how relevant a given prompt is to a specific topic (such as "gaming", "movies", "travel", "food", etc.).
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
        {format_instructions}
        Topic: "{topic}"
        Prompt: "{prompt}"
        Provide the relevance score and your reasoning in JSON format.
    """
    model_instance = ChatOllama(
        model=model_name, 
        temperature=0, 
        base_url=OLLAMA_BASE_URL,
        format="json"
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        user_template,
        input_variables=["topic", "prompt"], 
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = chat_prompt | model_instance | parser 
    
    try:
        generation = await chain.ainvoke(input={"topic": topic, "prompt": initial_prompt})
    except Exception as e:
        logger.error(f"Error during topic evaluation API call with Ollama: {e}", exc_info=True)
        await send_cot_message_to_websocket(config, f"Error during topic evaluation with Ollama: {e}")
        return {"is_completed": True, "error_message": f"Failed to evaluate topic relevance with Ollama: {str(e)}"}
        
    relevance_score = generation.get('relevance_score', 0.0) if isinstance(generation, dict) else 0.0
    await send_cot_message_to_websocket(config, f"Topic relevance score: {relevance_score:.2f}. Evaluation complete.")

    if relevance_score < 0.6:
        reasoning = generation.get('reasoning', 'The prompt has low relevance to the topic.') if isinstance(generation, dict) else 'The prompt has low relevance to the topic.'
        await send_cot_message_to_websocket(config, f"Relevance is too low ({relevance_score:.2f}). Stopping optimization. Reason: {reasoning}")
        return {"is_completed": True, "error_message": reasoning}
        
    return {}

# 번역 노드 (DeepL)
async def translation_prompt_node(status: MainState, config: dict) -> Dict[str, Any]:
    initial_prompt = status["initial_prompt"]
    await send_cot_message_to_websocket(config, "Translating the prompt to English for better AI understanding...")
    
    try:
        en_text = await text_translation(initial_prompt)
        await send_cot_message_to_websocket(config, "Prompt translation completed.")
        return {"translated_prompt": en_text}
    
    except Exception as e:
        logger.error(f"Error during prompt translation: {e}", exc_info=True)
        await send_cot_message_to_websocket(config, f"Error during prompt translation: {e}")
        return {"is_completed": True, "error_message": f"Failed to translate prompt: {str(e)}"}

# 프롬프트 평가 노드 (LLM-as-a-judge)
async def evaluate_prompt_node(status: MainState, config: dict) -> Dict[str, Any]:
    en_text = status.get("translated_prompt")
    if not en_text:
        logger.warning("Translated prompt is missing for evaluation.")
        await send_cot_message_to_websocket(config, "Error: Translated prompt is missing. Cannot proceed with evaluation.")
        return {"is_completed": True, "error_message": "Translated prompt is missing for evaluation."}

    model_name = status["model"]
    session_id = config.get("configurable", {}).get("session_id")

    await send_cot_message_to_websocket(config, "Starting prompt evaluation based on 6 main criteria (Clarity, Conciseness, etc.)...")
    
    current_weights = session_weights.get(str(session_id), {})
    if current_weights:
        await send_cot_message_to_websocket(config, f"Applying reinforcement learning weights for session '{session_id}'.")
    else:
        await send_cot_message_to_websocket(config, "No reinforcement learning weights found for this session. Using default weights (1.0).")

    en_all_criteria: List[str] = []
    flat_criteria_list: List[tuple[str, str, str]] = [] 
    for category, criteria_pairs in criteria.items():
        for kr_criterion, en_criterion in criteria_pairs:
            en_all_criteria.append(en_criterion)
            flat_criteria_list.append((category, kr_criterion, en_criterion))
    
    await send_cot_message_to_websocket(config, f"Calculating scores for a total of {len(en_all_criteria)} detailed items.")
    results_agg: Dict[str, Dict[str, Any]] = {category_name: {'criteria_results': [], 'average_score': 0.0} for category_name in criteria.keys()}
    
    try:
        api_result = await evaluate_text(en_text, en_all_criteria, model_name)
    except Exception as e:
        logger.error(f"Error during prompt evaluation (calling evaluate_text): {e}", exc_info=True)
        await send_cot_message_to_websocket(config, f"Error during call to evaluation function, stopping: {e}")
        return {"is_completed": True, "error_message": f"Error during prompt evaluation: {str(e)}"}
    
    if isinstance(api_result, dict) and 'error' in api_result:
        error_detail = api_result['error']
        logger.error(f"Error from evaluation function: {error_detail}")
        await send_cot_message_to_websocket(config, f"Evaluation failed: {error_detail}")
        return {"is_completed": True, "error_message": f"Prompt evaluation service error: {error_detail}"}
    
    evaluations_list = api_result.get('evaluations', [])
    if not isinstance(evaluations_list, list) or len(evaluations_list) != len(flat_criteria_list):
        logger.error(f"Mismatched evaluation results. Expected {len(flat_criteria_list)}, got {len(evaluations_list)}. API Response: {api_result}")
        await send_cot_message_to_websocket(config, "Incorrect format in evaluation results. Stopping.")
        return {"is_completed": True, "error_message": "Prompt evaluation result format error."}
    
    await send_cot_message_to_websocket(config, "Aggregating evaluation scores...")
    for i, (category, init_criterion, en_criterion) in enumerate(flat_criteria_list):
        eval_item = evaluations_list[i]
        if not isinstance(eval_item, dict):
            logger.warning(f"Invalid evaluation item format for '{en_criterion}': {eval_item}")
            raw_score = 0.0
        else:
            raw_score = float(eval_item.get("score", 0.0))
            
        weight = float(current_weights.get(str(i + 1), 1.0))
        final_score = raw_score * weight
        
        current_evaluation_text = "Satisfied" if final_score > 0.5 else "Not Satisfied"
        results_agg[category]['criteria_results'].append({
            'criterion': init_criterion, 'criterion_en': en_criterion,
            'score': float(final_score),
            'evaluation': current_evaluation_text
        })

    for category_name, category_data in results_agg.items():
        valid_scores = [item['score'] for item in category_data['criteria_results'] if isinstance(item.get('score'), (int, float))]
        if valid_scores:
            average_category_score_np = np.mean(valid_scores)
            results_agg[category_name]['average_score'] = float(average_category_score_np)
        else:
            results_agg[category_name]['average_score'] = 0.0
            
    await send_cot_message_to_websocket(config, "Prompt evaluation completed successfully.")
    return {"evaluation_data": results_agg}

# 프롬프트 제안 노드
async def improvement_prompt_node(status: MainState, config: dict) -> Dict[str, Any]:
    translated_prompt = status.get("translated_prompt")
    evaluation_data = status.get("evaluation_data")
    model_name = status["model"]

    if not translated_prompt or not evaluation_data:
        logger.warning("Missing translated prompt or evaluation data for improvement generation.")
        await send_cot_message_to_websocket(config, "Error: Missing data for generating improvements.")
        return {"is_completed": True, "error_message": "Missing data for improvement suggestions."}

    await send_cot_message_to_websocket(config, "Generating improvement ideas based on low-scoring items...")
    
    vulnerable_categories_names: List[str] = []
    IMPROVEMENT_THRESHOLD = 0.6 
    for category_data_val in evaluation_data.values():
        if isinstance(category_data_val, dict):
            average_score = category_data_val.get('average_score', 0.0)
            if average_score < IMPROVEMENT_THRESHOLD:
                for sub_category in category_data_val.get("criteria_results", []):
                    if isinstance(sub_category, dict) and sub_category.get("score", 0.0) < IMPROVEMENT_THRESHOLD:
                        criterion_en = sub_category.get("criterion_en")
                        if criterion_en:
                            vulnerable_categories_names.append(str(criterion_en))
    
    suggestions: Dict[str, str] = {}
    
    if vulnerable_categories_names:
        await send_cot_message_to_websocket(config, f"Generating suggestions for {len(vulnerable_categories_names)} items needing improvement.")
        tasks = [enhance_prompt(translated_prompt, vulnerable_category, model_name) for vulnerable_category in vulnerable_categories_names] 
        try:
            results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)
            for result_item in results_from_gather:
                if not isinstance(result_item, Exception) and isinstance(result_item, dict) and "error" not in result_item :
                    suggestions.update(result_item)
                elif isinstance(result_item, dict) and "error" in result_item:
                     logger.error(f"Error from enhance_prompt task: {result_item['error']}")
                elif isinstance(result_item, Exception):
                    logger.error(f"Exception in enhance_prompt task: {result_item}", exc_info=result_item)
        
        except Exception as e:
            logger.error(f"Error during asyncio.gather for enhance_prompt: {e}", exc_info=True)
            await send_cot_message_to_websocket(config, f"Error generating some improvement suggestions: {e}")
    else:
        await send_cot_message_to_websocket(config, "The prompt meets all criteria; no additional improvements are needed.")

    await send_cot_message_to_websocket(config, "Prompt improvement idea generation completed.")
    return {"improvement_suggestions": suggestions}

# 프롬프트 보완 노드
async def enhance_prompt_node(status: MainState, config: dict) -> Dict[str, Any]:
    websocket = config.get("configurable", {}).get("websocket")
    session_id = config.get("configurable", {}).get("session_id", "unknown_session")
    
    await send_cot_message_to_websocket(config, "Waiting for user feedback... (Max 120 seconds)")
    human_feedback_text: Optional[str] = await session_manager.wait_for_feedback(str(session_id), timeout=120.0)
    
    if human_feedback_text:
        await send_cot_message_to_websocket(config, "User feedback received. Incorporating into the final prompt.")
    else:
        await send_cot_message_to_websocket(config, "No user feedback received. Generating final prompt with existing information.")
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"type": "hitl_error", "text": "No human feedback entered within the time limit."})
            except Exception as e:
                logger.warning(f"Failed to send HITL error to WebSocket: {e}", exc_info=True)

    await send_cot_message_to_websocket(config, "Generating the final prompt by combining all analysis results...")
    
    english_context_prompt = status.get("translated_prompt")
    improvement_suggestions = status.get("improvement_suggestions", {})
    model_name = status["model"]
    
    if not english_context_prompt:
        error_msg = "Error: Translated prompt is missing, cannot generate final prompt."
        logger.error(error_msg)
        await send_cot_message_to_websocket(config, error_msg)
        return {"optimized_prompt": error_msg, "is_completed": True, "error_message": error_msg}
        
    combined_enhancements = dict(improvement_suggestions) if improvement_suggestions else {}
    if human_feedback_text:
        combined_enhancements["User Additional Feedback (HITL)"] = human_feedback_text

    korean_optimized_prompt = ""
    try:
        stream_type = "optimize" 
        async for chunk in generate_final_prompt(english_context_prompt, combined_enhancements, model_name):
            korean_optimized_prompt += chunk
            if websocket and websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json({"type": stream_type, "text": chunk})
                except Exception as e:
                    logger.warning(f"Failed to stream chunk to WebSocket: {e}", exc_info=True)
                    
    except Exception as e:
        error_message = f"Error during final prompt generation streaming: {e}"
        logger.error(error_message, exc_info=True)
        await send_cot_message_to_websocket(config, error_message)
        return {"optimized_prompt": f"Error: {str(e)}", "is_completed": True, "error_message": error_message}

    await send_cot_message_to_websocket(config, "Final prompt generation completed.")
    return {
        "human_feedback": human_feedback_text if human_feedback_text else None, 
        "optimized_prompt": korean_optimized_prompt, 
        "is_completed": False 
    }

# 에러 출력 노드
async def stream_error_node(status: MainState, config: dict) -> Dict[str, Any]:
    error_message = status.get("error_message", "An unspecified error occurred.")
    logger.error(f"Error node triggered: {error_message}")
    await send_cot_message_to_websocket(config, f"An error occurred: {error_message}")
    
    websocket = config.get("configurable", {}).get("websocket")
    if websocket and websocket.client_state == WebSocketState.CONNECTED:
        try:
            await websocket.send_json({"type": "error", "text": error_message})
        except Exception as e:
            logger.error(f"Failed to send error message via WebSocket: {e}", exc_info=True)
    return status 

# Human in the loop 요청 노드
async def hitl_request_node(status: MainState, config: dict) -> Dict[str, Any]:
    model_name = status["model"] 
    prompt_text = status["initial_prompt"] 
    websocket = config.get("configurable", {}).get("websocket")

    if not websocket or websocket.client_state != WebSocketState.CONNECTED:
        logger.warning("HITL request: WebSocket not connected. Skipping.")
        return {}

    try:
        await send_cot_message_to_websocket(config, "Generating a question for user feedback (Human-in-the-Loop)...")
        hitl_response = await human_in_the_loop_prompt(text=prompt_text, model_name=model_name) 
        
        if "error" in hitl_response:
            hitl_question = f"Could not generate HITL question: {hitl_response['error']}"
            logger.error(hitl_question)
        else:
            hitl_question = hitl_response.get("question", "Could not generate a specific question at this time.")

        await websocket.send_json({"type": "hitl", "text": hitl_question})
        await send_cot_message_to_websocket(config, "Question for user feedback sent.")
        return {'human_feedback_ai': hitl_question}
    
    except Exception as e:
        logger.error(f"Error in HITL request node: {e}", exc_info=True)
        await send_cot_message_to_websocket(config, f"Error generating HITL question: {e}")
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"type": "error", "text": f"Failed to generate HITL question: {str(e)}"})
            except Exception as ws_e:
                logger.error(f"Failed to send HITL error to WebSocket: {ws_e}", exc_info=True)
        return {}

# ------------------------------
#   Tree of Toughts
# ------------------------------

# Tree of Thoughts 생각 서브 함수
async def async_invoke_llm_for_tot(llm: ChatOllama, system_prompt: str, human_prompt: str, temperature: float = 0.7) -> str:
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ], config={"temperature": temperature}) 
        return response.content
    
    except Exception as e:
        error_message = f"LLM asynchronous call error: {e}"
        logger.error(error_message, exc_info=True)
        return error_message

# Tree of Thoughts 생각 평가 함수
async def evaluate_thought_with_llm_for_tot(llm_judge: ChatOllama, thought_text: str, main_prompt: str) -> tuple[float, str]:
    system_prompt = (
        "You are an impartial evaluator AI that rates the quality of a 'thought' based on given criteria and assigns a score. "
        "The score must be an integer between 0 and 100. "
        "Evaluation criteria are: "
        "1. Relevance to the main goal (main_prompt), "
        "2. Logical consistency and clarity, "
        "3. Originality and insight, "
        "4. Potential contribution to problem-solving or goal achievement. "
        "Your evaluation result MUST be in the following format ONLY:\n"
        "Score: [integer between 0-100]\n"
        "Reasoning: [brief evaluation reasoning]"
    )
    human_prompt = (
        f"Main Goal: '{main_prompt}'\n\n"
        f"Thought to Evaluate: '{thought_text}'\n\n"
        "Provide the score and evaluation reasoning for this thought according to the criteria and format above."
    )
    
    raw_evaluation = await async_invoke_llm_for_tot(llm_judge, system_prompt, human_prompt, temperature=0.2)

    score = 0.0
    reasoning = "Evaluation failed or format error"
    try:
        score_match = re.search(r"Score:\s*(\d+)", raw_evaluation, re.IGNORECASE)
        reasoning_match = re.search(r"Reasoning:\s*(.+)", raw_evaluation, re.IGNORECASE | re.DOTALL)
        
        if score_match:
            score_val = int(score_match.group(1))
            score = float(max(0, min(100, score_val)))
        else:
            logger.warning(f"Could not parse score from LLM evaluation. Response: {raw_evaluation}")

        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            logger.warning(f"Could not parse reasoning from LLM evaluation. Response: {raw_evaluation}")
            if score_match: reasoning = "Reasoning parsing failed, score extracted."

    except Exception as e:
        logger.error(f"Error parsing LLM evaluation result: {e}. Original response: {raw_evaluation}", exc_info=True)
    return score, reasoning

# Tree of Thoughts 초기화 노드 
async def dr_initializer_node(state: MainState) -> Dict[str, Any]:
    logger.info("Initializing Detailed Report (ToT) state...")
    report_state: ReportState = {
        "dr_tavily_initial_context": "No initial context information available.\n",
        "dr_active_thoughts": [],
        "dr_all_generated_thoughts": [],
        "dr_selected_best_thought": None,
        "dr_current_depth": 0,
        "dr_max_depth": 1,
        "dr_thoughts_per_expansion": 2, 
        "dr_top_k_to_keep": 1, 
        "dr_generated_for_evaluation": [],
        "detailed_markdown_report": None,
    }

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    original_prompt_for_search = state.get('optimized_prompt', state.get('initial_prompt'))

    if tavily_api_key and original_prompt_for_search:
        try:
            logger.debug("Attempting Tavily search for initial context...")
            tavily_client = TavilyClient(api_key=tavily_api_key)
            search_results = await asyncio.to_thread(
                tavily_client.search, 
                query=original_prompt_for_search, 
                search_depth="basic", 
                max_results=3
            )
            
            formatted_initial_search = f"Initial context information for '{escape_curly_braces(original_prompt_for_search)}' (Tavily Search):\n"
            if search_results and search_results.get("results"):
                for res in search_results["results"]:
                    title = escape_curly_braces(res.get('title', 'N/A'))
                    url = res.get('url', 'N/A')
                    content = escape_curly_braces(res.get('content', 'No content'))
                    formatted_initial_search += f"- Source: {title} ({url})\n  Content: {content}\n\n"
                logger.debug("Tavily search successful, context formatted.")
            else:
                formatted_initial_search += "No results found.\n"
                logger.debug("Tavily search yielded no results.")
            report_state['dr_tavily_initial_context'] = formatted_initial_search

        except Exception as e:
            logger.error(f"Error during Tavily initial search: {e}", exc_info=True)
            report_state['dr_tavily_initial_context'] = "Error during Tavily initial search.\n"
    elif not tavily_api_key:
        logger.warning("TAVILY_API_KEY not set. Skipping Tavily search for initial context.")
    elif not original_prompt_for_search:
        logger.warning("Original prompt for search is empty. Skipping Tavily search.")

    root_thought_text = f"Initial considerations for '{original_prompt_for_search}'"
    root_thought_id = str(uuid.uuid4())
    root_thought: Thought = {
        "id": root_thought_id, "text": root_thought_text, "score": 50.0, "reasoning": "Initial thought, default score",
        "depth": 0, "parent_id": None, "path_string": root_thought_text[:30]+"..."
    }
    report_state['dr_active_thoughts'] = [root_thought]
    report_state.setdefault('dr_all_generated_thoughts', []).append(root_thought)
    report_state['dr_selected_best_thought'] = root_thought
    logger.info("Detailed Report (ToT) state initialized.")
    return {"report_data": report_state}

# Tree of Thoughts 생각 생성 노드
async def dr_thought_generator_node(state: MainState) -> Dict[str, Any]:
    report_state = state["report_data"].copy() if state.get("report_data") else {}
    if not report_state:
        logger.error("Report data is missing in dr_thought_generator_node.")
        return {"report_data": {"dr_generated_for_evaluation": []}}

    model_name = state['model']
    newly_generated_thoughts: List[Thought] = []
    optimized_prompt = state.get('optimized_prompt', state.get('initial_prompt', ''))

    if not report_state.get('dr_active_thoughts'):
        logger.info("No active thoughts to expand. Skipping thought generation.")
        report_state['dr_generated_for_evaluation'] = []
        return {"report_data": report_state}
    
    try:
        llm_generator = ChatOllama(model=model_name, temperature=0.7, base_url=OLLAMA_BASE_URL)
    except Exception as e:
        logger.error(f"Failed to initialize Ollama LLM generator: {e}", exc_info=True)
        report_state['dr_generated_for_evaluation'] = []
        return {"report_data": report_state}

    initial_context = report_state.get('dr_tavily_initial_context', "No initial context information.")
    current_depth_for_expansion = report_state.get('dr_current_depth', 0)

    for parent_thought in report_state.get('dr_active_thoughts', []):
        if parent_thought['depth'] != current_depth_for_expansion:
            continue
        
        logger.debug(f"Expanding from parent thought (ID: {parent_thought['id'][:4]}, Depth: {parent_thought['depth']}): '{parent_thought['text'][:50]}...' (Score: {parent_thought.get('score', 0.0):.2f})")
        
        generation_tasks = []
        for i in range(report_state.get('dr_thoughts_per_expansion', 1)):
            system_prompt = "You are a helpful AI assistant that generates concise and insightful next-step thoughts or alternative perspectives on complex problems. Each thought should be a single, coherent idea that can be evaluated. All responses must be in Korean."
            human_prompt_content = (
                f"Here is some relevant initial context:\n{initial_context}\n\n"
                f"Overall Goal: '{optimized_prompt}'\n"
                f"Current Thought: '{parent_thought['text']}'\n\n"
                "Based on the information above, generate ONE clear and specific next-step thought or an alternative approach to consider. Keep it concise (1-2 sentences). Your response must be in Korean."
            )
            generation_tasks.append(async_invoke_llm_for_tot(llm_generator, system_prompt, human_prompt_content))
        
        try:
            generated_texts = await asyncio.gather(*generation_tasks)
        except Exception as e:
            logger.error(f"Error during thought generation gather for parent '{parent_thought['text'][:20]}...': {e}", exc_info=True)
            generated_texts = ["[Error during generation]" for _ in generation_tasks]

        for gen_idx, generated_text in enumerate(generated_texts):
            new_thought_id = str(uuid.uuid4())
            new_thought: Thought = {
                "id": new_thought_id, "text": generated_text, "score": 0.0, "reasoning": None,
                "depth": parent_thought['depth'] + 1, "parent_id": parent_thought['id'],
                "path_string": f"{parent_thought.get('path_string','Root')} -> {generated_text[:30]}..."
            }
            newly_generated_thoughts.append(new_thought)
            logger.debug(f"  + Generated (Parallel {gen_idx+1}): '{new_thought['text'][:60]}...' (Depth: {new_thought['depth']})")

    report_state['dr_generated_for_evaluation'] = newly_generated_thoughts
    logger.info(f"Generated {len(newly_generated_thoughts)} new thoughts for evaluation.")
    return {"report_data": report_state}

# Tree of Thoughts 생각 평가 노드
async def dr_evaluator_node(state: MainState) -> Dict[str, Any]:
    report_state = state["report_data"].copy() if state.get("report_data") else {}
    if not report_state:
        logger.error("Report data is missing in dr_evaluator_node.")
        return {"report_data": {}} 

    model_name = state['model']
    optimized_prompt = state.get('optimized_prompt', state.get('initial_prompt', ''))
    
    evaluated_thoughts_for_current_depth: List[Thought] = []
    thoughts_to_evaluate = report_state.get('dr_generated_for_evaluation', [])

    if not thoughts_to_evaluate:
        logger.info("No thoughts to evaluate in this step.")
        report_state['dr_active_thoughts'] = []
        return {"report_data": report_state}
    
    logger.info(f"Evaluating {len(thoughts_to_evaluate)} thoughts...")
    try:
        llm_judge = ChatOllama(model=model_name, temperature=0.2, base_url=OLLAMA_BASE_URL)
    except Exception as e:
        logger.error(f"Failed to initialize Ollama LLM judge: {e}", exc_info=True)
        for thought_draft in thoughts_to_evaluate:
            evaluated_thought = {**thought_draft, 'score': 0.0, 'reasoning': "Judge LLM initialization failed"}
            evaluated_thoughts_for_current_depth.append(evaluated_thought)
        report_state['dr_active_thoughts'] = [] 
        report_state.setdefault('dr_all_generated_thoughts', []).extend(evaluated_thoughts_for_current_depth)
        return {"report_data": report_state}

    evaluation_tasks = []
    for thought_draft in thoughts_to_evaluate:
        evaluation_tasks.append(evaluate_thought_with_llm_for_tot(llm_judge, thought_draft['text'], optimized_prompt))

    try:
        evaluation_results = await asyncio.gather(*evaluation_tasks)
    except Exception as e:
        logger.error(f"Error during gather for thought evaluations: {e}", exc_info=True)
        evaluation_results = [(0.0, "Evaluation failed due to gather error") for _ in evaluation_tasks]

    for i, thought_draft in enumerate(thoughts_to_evaluate):
        score, reasoning = evaluation_results[i]
        evaluated_thought = thought_draft.copy()
        evaluated_thought['score'] = score
        evaluated_thought['reasoning'] = reasoning
        evaluated_thoughts_for_current_depth.append(evaluated_thought)
        report_state.setdefault('dr_all_generated_thoughts', []).append(evaluated_thought)
        logger.debug(f"  Evaluated thought (ID: {evaluated_thought['id'][:4]}): '{evaluated_thought['text'][:50]}...' Score: {score:.2f}, Reasoning: {reasoning[:30]}...")
        
    evaluated_thoughts_for_current_depth.sort(key=lambda t: t.get('score', 0.0), reverse=True)
    top_k_to_keep = report_state.get('dr_top_k_to_keep', 1)
    report_state['dr_active_thoughts'] = evaluated_thoughts_for_current_depth[:top_k_to_keep]
    
    if evaluated_thoughts_for_current_depth:
        current_best_in_batch = evaluated_thoughts_for_current_depth[0]
        if report_state.get('dr_selected_best_thought') is None or \
           current_best_in_batch.get('score', 0.0) > report_state.get('dr_selected_best_thought', {}).get('score', -1.0):
            report_state['dr_selected_best_thought'] = current_best_in_batch
            logger.info(f"New best thought selected (ID: {current_best_in_batch['id'][:4]}, Score: {current_best_in_batch.get('score',0.0):.2f})")
            
    elif not report_state.get('dr_active_thoughts') and report_state.get('dr_selected_best_thought') is None and report_state.get('dr_all_generated_thoughts'):
        all_thoughts = report_state.get('dr_all_generated_thoughts', [])
        if all_thoughts:
            sorted_all = sorted(all_thoughts, key=lambda t: t.get('score', 0.0), reverse=True)
            if sorted_all: 
                report_state['dr_selected_best_thought'] = sorted_all[0]
                logger.info(f"Fallback best thought selected from all generated (ID: {sorted_all[0]['id'][:4]}, Score: {sorted_all[0].get('score',0.0):.2f})")
    
    report_state['dr_generated_for_evaluation'] = []
    logger.info(f"Evaluation complete. Active thoughts for next depth: {len(report_state.get('dr_active_thoughts',[]))}")
    return {"report_data": report_state}

# Tree of Thoughts 컨트롤 노드
async def dr_tot_control_node(state: MainState) -> Dict[str, Any]:
    report_state = state["report_data"].copy() if state.get("report_data") else {}
    if not report_state:
        logger.error("Report data is missing in dr_tot_control_node.")
        return {"report_data": {}}

    current_depth = report_state.get('dr_current_depth', -1) + 1
    report_state['dr_current_depth'] = current_depth
    
    logger.info(f"--- ToT Control: Advancing to Depth {current_depth} ---")
    active_thoughts_for_logging = report_state.get('dr_active_thoughts', [])
    if active_thoughts_for_logging:
        logger.info(f"Active thoughts for depth {current_depth} (Top {len(active_thoughts_for_logging)}):")
        for i, thought in enumerate(active_thoughts_for_logging):
            logger.info(f"  {i+1}. (ID: {thought.get('id','N/A')[:4]}, D: {thought.get('depth','N/A')}) '{thought.get('text','N/A')[:60]}...' (S: {thought.get('score',0.0):.2f}, R: {thought.get('reasoning', 'N/A')[:30]}...)")
    else:
        logger.info(f"No active thoughts to proceed to depth {current_depth}.")
        
    return {"report_data": report_state}

# 프롬프트 해킹 방지 함수
def escape_curly_braces(text: Optional[str]) -> str:
    if text is None:
        return ""
    return str(text).replace("{", "{{").replace("}", "}}")

# Tree of Thoughts 리포트 생성 노드
async def dr_markdown_report_generator_node(state: MainState, config: dict) -> Dict[str, Any]:
    websocket = config.get("configurable", {}).get("websocket")
    report_state = state.get("report_data", {}).copy()
    model_name = state['model']
    best_thought_data = report_state.get('dr_selected_best_thought')
    original_prompt_for_report = state.get('optimized_prompt', state.get('initial_prompt'))

    logger.info("Starting detailed markdown report generation...")

    if not websocket or websocket.client_state != WebSocketState.CONNECTED:
        logger.warning("WebSocket not connected. Report streaming will not occur.")

    if not best_thought_data or not original_prompt_for_report:
        error_msg = "Failed to generate detailed report: Missing essential data (best thought or original prompt)."
        logger.error(error_msg)
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"type": "error", "text": error_msg})
            except Exception as e:
                logger.warning(f"Failed to send error to WebSocket: {e}", exc_info=True)
        report_state['detailed_markdown_report'] = error_msg
        return {"report_data": report_state, "is_completed": True, "error_message": error_msg}

    escaped_original_prompt = escape_curly_braces(original_prompt_for_report)
    best_thought_text_val = best_thought_data.get('text') if isinstance(best_thought_data, dict) else None
    escaped_best_thought_text = escape_curly_braces(best_thought_text_val) if best_thought_text_val else "No core idea analyzed."

    tavily_search_context_for_prompt = "No web search information available.\n"
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        try:
            tavily_client = TavilyClient(api_key=tavily_api_key)
            search_query_best_thought = best_thought_text_val if best_thought_text_val else ''
            search_query = f"In-depth analysis report for '{escaped_original_prompt}' and '{escaped_best_thought_text}'"
            
            logger.debug(f"Performing Tavily web search... (Query: {search_query[:70]}...)")
            search_results = await asyncio.to_thread(
                tavily_client.search, query=search_query, search_depth="advanced", max_results=5
            )
            
            if search_results and search_results.get("results"):
                temp_tavily_context = f"Web search results for '{escape_curly_braces(search_query)}':\n\n"
                for res_idx, res in enumerate(search_results["results"]):
                    title = escape_curly_braces(res.get('title', 'N/A'))
                    url = res.get('url', 'N/A') 
                    content = escape_curly_braces(res.get('content', 'No content'))
                    temp_tavily_context += f"Source {res_idx+1}: {title} ({url})\nContent: {content}\n\n"
                tavily_search_context_for_prompt = temp_tavily_context
                logger.debug("Tavily search successful for report.")
            else:
                tavily_search_context_for_prompt = "No relevant web search results found.\n"
                logger.debug("Tavily search yielded no results for report.")
        except Exception as e:
            logger.warning(f"Error during Tavily search for report: {e}", exc_info=True)
            tavily_search_context_for_prompt = "An error occurred during web search.\n"
    else:
        logger.warning("TAVILY_API_KEY not set. Skipping web search for report.")

    llm_report_writer = ChatOllama(
        model=model_name, 
        temperature=0.7, 
        streaming=True, 
        base_url=OLLAMA_BASE_URL
    )
    
    system_prompt = """
    You are an AI writer specializing in creating professional analytical reports. 
    Based on the given topic, core idea, and web search results, you must generate a well-structured and in-depth markdown report.
    The report MUST follow this structure: 'Title', 'Introduction', 'Table of Contents', 'Body (for each table of contents item)', 'Conclusion'.
    Each body section should actively utilize the provided web search results to present rich and detailed content.
    All output MUST be in Korean. Output only the report content; do not include any other extraneous explanations.
    """

    human_prompt = f"""
    Please generate a complete detailed report in markdown format based on the following information.

    ### 1. Basic Topic of the Report (Original Prompt)
    "{escaped_original_prompt}"

    ### 2. Core Idea (Result of ToT)
    "{escaped_best_thought_text}"

    ### 3. Reference Material (Web Search Results)
    {tavily_search_context_for_prompt}
    
    ### 4. Report Writing Guidelines and Output Format
    - Title: Create a concise and informative title for the report.
    - Introduction: Briefly introduce the purpose and scope of the report.
    - Table of Contents: Generate a clickable table of contents (e.g., using markdown links like `[Section 1](#section-1)`).
    - Body: Divide the main content into logical sections based on the table of contents. Each section should be detailed, well-reasoned, and incorporate information from the web search results where relevant. Use markdown formatting for headings, lists, bold text, etc.
    - Conclusion: Summarize the key findings and offer a final perspective.
    - Language: The entire report must be in Korean.
    Ensure the report is comprehensive, coherent, and professionally written.
    """

    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", human_prompt)])
    chain = prompt_template | llm_report_writer | StrOutputParser()

    full_report = ""
    logger.info("Streaming detailed report to WebSocket using Ollama...")
    try:
        async for chunk in chain.astream({}):
            full_report += chunk
            if websocket and websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json({"type": "result", "text": chunk})
                except Exception as e:
                    logger.warning(f"Failed to stream report chunk to WebSocket: {e}", exc_info=True)
        logger.info("Detailed report streaming with Ollama complete.")
    
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected during report streaming.")
        report_state['detailed_markdown_report'] = full_report
        return {"report_data": report_state, "is_completed": True, "error_message": "WebSocket disconnected during report generation."}
    
    except Exception as e:
        logger.error(f"Error during report generation streaming with Ollama: {e}", exc_info=True)
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"type": "error", "text": "A server error occurred during report generation."})
            except Exception as ws_e:
                logger.error(f"Failed to send report error to WebSocket: {ws_e}", exc_info=True)
        report_state['detailed_markdown_report'] = full_report
        return {"report_data": report_state, "is_completed": True, "error_message": f"Error generating report with Ollama: {str(e)}"}

    report_state['detailed_markdown_report'] = full_report.strip()
    logger.info("Detailed markdown report generation with Ollama finished successfully.")
    
    return {"report_data": report_state, "is_completed": True}

