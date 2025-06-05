import os
import logging
import asyncio
import deepl
from typing import List, Dict, Any, AsyncGenerator, Optional

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from app.model.schema import EvaluationResponse
from app.log import setup_logging
from dotenv import load_dotenv

load_dotenv()

logger: logging.Logger = setup_logging()

DEEPL_API_KEY: Optional[str] = os.getenv("DEEPL_API_KEY")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
translator: Optional[deepl.Translator] = None

if DEEPL_API_KEY:
    try:
        translator = deepl.Translator(auth_key=DEEPL_API_KEY)
        logger.info("DeepL Translator initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize DeepL Translator: {e}", exc_info=True)
        translator = None
else:
    logger.warning("DEEPL_API_KEY not found. Text translation will not be available.")


async def text_translation(text: str, target_lang: str = "EN-US") -> str:
    """Translates text to the target language using DeepL."""
    if not translator:
        logger.warning("DeepL Translator is not available. Returning original text.")
        return text
    if not text or not text.strip():
        logger.debug("Input text for translation is empty. Returning as is.")
        return text
    try:
        logger.debug(f"Attempting to translate text to {target_lang}: '{text[:50]}...'")
        result = await asyncio.to_thread(translator.translate_text, text=text, target_lang=target_lang)
        translated_text = str(result).strip()
        logger.info(f"Text translated successfully to {target_lang}.")
        return translated_text
    except deepl.DeepLException as e:
        logger.error(f"DeepL API error during translation: {e}", exc_info=True)
        return text 
    except Exception as e:
        logger.error(f"An unexpected error occurred during text translation: {e}", exc_info=True)
        return text


async def evaluate_text(text: str, criteria_list: List[str], model_name: str = "llama3") -> Dict[str, Any]:
    """Evaluates text against a list of criteria using an Ollama LLM."""
    if not text or not text.strip():
        logger.warning("Text for evaluation is empty.")
        return {"error": "Text for evaluation cannot be empty."}
    if not criteria_list:
        logger.warning("Criteria list for evaluation is empty.")
        return {"error": "Criteria list for evaluation cannot be empty."}

    logger.debug(f"Evaluating text against {len(criteria_list)} criteria using Ollama model {model_name} at {OLLAMA_BASE_URL}.")
    criteria_string = "\n".join(f'{idx}. "{criterion}"' for idx, criterion in enumerate(criteria_list, start=1))
    
    parser = JsonOutputParser(pydantic_object=EvaluationResponse)

    system_prompt_template = (
        "You are an expert text evaluator.\n"
        "Your task is to evaluate a given text against a list of criteria.\n"
        "For each criterion, determine if the text meets the criterion and provide a confidence score "
        "between 0.0 and 1.0 (where 1.0 means a perfect match and 0.0 means no match).\n"
        "Your response MUST be a valid JSON object. The JSON object should have a single key \"evaluations\".\n"
        "The value of \"evaluations\" should be a list of objects, where each object corresponds to one "
        "criterion from the input list (in the same order) and has two keys:\n"
        "1. \"criterion\": The exact criterion string.\n"
        "2. \"score\": A float value between 0.0 and 1.0 representing the confidence score.\n"
        "Example for one criterion: {{\"criterion\": \"example criterion text\", \"score\": 0.81}}"
    )

    user_prompt_template = (
        "Please evaluate the following text:\n"
        "{format_instructions}\n"
        "--- TEXT START ---\n"
        "{text}\n"
        "--- TEXT END ---\n\n"
        "Against these criteria:\n"
        "--- CRITERIA START ---\n"
        "{criteria_string}\n"
        "--- CRITERIA END ---\n\n"
    )
    
    system_message = SystemMessagePromptTemplate.from_template(system_prompt_template)
    human_message = HumanMessagePromptTemplate.from_template(
        user_prompt_template, 
        input_variables=["text", "criteria_string"], 
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    
    model = ChatOllama(
        model=model_name,
        temperature=0,
        base_url=OLLAMA_BASE_URL,
        format="json"
    )
    
    chain = chat_prompt | model | parser
    
    try:
        logger.debug("Invoking Ollama LLM for text evaluation.")
        result = await chain.ainvoke({"text": text, "criteria_string": criteria_string})
        logger.info("Text evaluation with Ollama completed successfully.")
        if isinstance(result, EvaluationResponse):
            return result.dict()
        return result 
        
    except Exception as e:
        logger.error(f"Text evaluation with Ollama failed: {e}", exc_info=True)
        return {"error": f"Evaluation with Ollama failed: {str(e)}"}


async def enhance_prompt(existing_prompt: str, improvement_points: str, model_name: str = "llama3") -> Dict[str, Any]:
    """Enhances an existing prompt based on specified improvement points using an Ollama LLM."""
    if not existing_prompt or not existing_prompt.strip():
        logger.warning("Existing prompt for enhancement is empty.")
        return {"error": "Existing prompt cannot be empty."}
    if not improvement_points or not improvement_points.strip():
        logger.warning("Improvement points for enhancement are empty.")
        return {"error": "Improvement points cannot be empty."}

    logger.debug(f"Enhancing prompt with points: '{improvement_points[:50]}...' using Ollama model {model_name} at {OLLAMA_BASE_URL}.")
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

    model = ChatOllama(
        model=model_name, 
        temperature=0.7,
        base_url=OLLAMA_BASE_URL
    )
    chain = prompt | model | StrOutputParser()

    try:
        logger.debug("Invoking Ollama LLM for prompt enhancement.")
        response = await chain.ainvoke({
            "existing_prompt": existing_prompt,
            "improvement_points": improvement_points
        })
        logger.info("Prompt enhancement with Ollama completed successfully.")
        return {improvement_points: response}
        
    except Exception as e:
        logger.error(f"Error enhancing prompt with Ollama: {e}", exc_info=True)
        return {"error": f"Error enhancing prompt with Ollama: {str(e)}"}


async def human_in_the_loop_prompt(text: str, model_name: str = "llama3") -> Dict[str, Any]:
    """Generates questions to clarify ambiguities for a given text using an Ollama LLM."""
    if not text or not text.strip():
        logger.warning("Text for HITL question generation is empty.")
        return {"error": "Text for HITL prompt generation cannot be empty."}

    logger.debug(f"Generating HITL question for text: '{text[:50]}...' using Ollama model {model_name} at {OLLAMA_BASE_URL}.")
    
    system_prompt_content = (
        "You're an expert with sharp insights. Your job is to identify hidden context or scarce information "
        "in a given text to generate clear and concise questions for clarification.\n\n"
        "For example, when a user asks “how to set environment variables,” you might want to know what "
        "operating system they are using (Windows, macOS, Linux, etc.).\n\n"
        "You can create one or multiple questions. The answer should consolidate all generated questions "
        "into a single, natural-sounding sentence or question.\n\n"
        "Example: If questions like “Are you using Windows OS or macOS?” and “Do you need examples?” are generated, "
        "the output should be something like “Are you using Windows OS or macOS, and do you need examples of setting environment variables?”.\n\n"
        "When outputting messages, match the language of the user's input text. "
        "For example, if the user entered text in Korean, output your question in Korean."
    )
    
    user_prompt_content = (
        "Please generate a clarifying question for the following text:\n"
        "--- TEXT START ---\n"
        "{text}\n"
        "--- TEXT END ---"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_content),
        ("user", user_prompt_content)
    ])
    
    model = ChatOllama(
        model=model_name, 
        temperature=0.6,
        base_url=OLLAMA_BASE_URL
    )
    chain = prompt | model | StrOutputParser()
    
    try:
        logger.debug("Invoking Ollama LLM for HITL question generation.")
        response = await chain.ainvoke({"text": text})
        logger.info("HITL question generation with Ollama completed successfully.")
        return {"question": response}
        
    except Exception as e:
        logger.error(f"HITL question generation with Ollama failed: {e}", exc_info=True)
        return {"error": f"HITL question generation with Ollama failed: {str(e)}"}

    
async def generate_final_prompt(original_english_prompt: str, enhancement_results: Dict[str, str], model_name: str = "llama3") -> AsyncGenerator[str, None]:
    """
    Generates a final, enhanced prompt in Korean using an Ollama LLM,
    integrating the original English prompt with improvement suggestions.
    """
    if not original_english_prompt or not original_english_prompt.strip():
        logger.error("Original English prompt is empty for final prompt generation.")
        yield "Error: Original English prompt cannot be empty."
        return

    logger.debug(f"Generating final Korean prompt from English prompt: '{original_english_prompt[:50]}...' and {len(enhancement_results)} enhancements, using Ollama model {model_name} at {OLLAMA_BASE_URL}.")

    enhancements_text = ""
    for idx, (category, suggestion) in enumerate(enhancement_results.items(), 1):
        enhancements_text += f"\nImprovement {idx} - Category/Source: {category}\nSuggestion: {suggestion}\n"

    system_template = """
    You are a prompt engineering expert. Your task is to integrate an original prompt (provided for context) 
    with multiple improvement suggestions to create a final, enhanced prompt.
    Carefully analyze the original prompt and all provided improvement suggestions.
    Then, create a comprehensive final prompt that incorporates all valuable aspects of the improvements 
    while maintaining consistency and clarity.

    IMPORTANT: The final enhanced prompt MUST be in KOREAN. 
    Do not provide any explanations, notes, or text outside of the KOREAN prompt itself.
    """
    user_template = """
    Original Prompt (English, for context only):
    {original_english_prompt}

    Improvement Suggestions (based on the original prompt, may include user feedback):
    {enhancements_text}

    If the "Improvement Suggestions" section contains an item with "사용자 추가 피드백 (HITL)" or "User Additional Feedback (HITL)" 
    as its category/source, please prioritize this feedback. This means if the HITL feedback contradicts 
    other information or the original prompt, the HITL feedback should take precedence for that specific aspect.

    For example, if the original prompt implied "Windows OS", but the HITL feedback states "I'm actually using a Mac OS", 
    the final prompt should reflect the use of "Mac OS".

    Please write a final, enhanced prompt in KOREAN that effectively integrates all valuable improvements.
    Provide ONLY the KOREAN prompt.
    """
    prompt_template_messages = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])

    model_instance = ChatOllama(
        model=model_name,
        temperature=0.5,
        streaming=True,
        base_url=OLLAMA_BASE_URL
    )
    chain = prompt_template_messages | model_instance | StrOutputParser()

    logger.debug("Streaming final Korean prompt generation with Ollama.")
    try:
        async for korean_chunk in chain.astream({
            "original_english_prompt": original_english_prompt,
            "enhancements_text": enhancements_text
        }):
            if korean_chunk:
                yield korean_chunk
        logger.info("Final Korean prompt streaming with Ollama completed.")
    except Exception as e:
        logger.error(f"Error during final prompt streaming with Ollama: {e}", exc_info=True)
        yield f"\nError during final prompt generation with Ollama: {str(e)}"
