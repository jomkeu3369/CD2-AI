import os
import deepl

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from app.model.schema import EvaluationResponse
from dotenv import load_dotenv
from typing import List, Dict, Any, AsyncGenerator

load_dotenv()
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
translator = deepl.Translator(auth_key=DEEPL_API_KEY)

# 번역
async def text_translation(text:str, end_lang:str="EN-US") -> str:
    try:
        result = translator.translate_text(text=text, target_lang=end_lang)
        return str(result).strip()
    except:
        return text

# 평가 분류
async def evaluate_text(text: str, criteria_list: List[str], model_name: str = "gpt-4o-mini") -> Dict[str, Any]:

    criteria_string = "\n".join(f'{idx}. "{criterion}"' for idx, criterion in enumerate(criteria_list, start=1))
    
    parser = JsonOutputParser(pydantic_object=EvaluationResponse)

    system_prompt = (
        "You are an expert text evaluator.\n"
        "Your task is to evaluate a given text against a list of criteria.\n"
        "For each criterion, determine if the text meets the criterion and provide a confidence score "
        "between 0.0 and 1.0 (where 1.0 means a perfect match and 0.0 means no match).\n"
        "Your response MUST be a valid JSON object. The JSON object should have a single key \"evaluations\".\n"
        "The value of \"evaluations\" should be a list of objects, where each object corresponds to one "
        "criterion from the input list (in the same order) and has three keys:\n"
        "1. \"criterion\": The exact criterion string.\n"
        "2. \"score\": A float value between 0.0 and 1.0 representing the confidence score.\n"
        "Example for one criterion: \"criterion\": \"example criterion text\", \"score\": 0.81"
    )

    user_prompt = (
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
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        user_prompt, 
        input_variables=["text", "criteria_string"], 
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            system_message_prompt,
            human_message_prompt
        ]
    )
    
    model = ChatOpenAI(
        model=model_name,
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    chain = prompt | model | JsonOutputParser()
    
    try:
        result = await chain.ainvoke({"text": text, "criteria_string": criteria_string})
        return result
        
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

async def human_in_the_loop_prompt(text: str, model_name: str = "gpt-4o-mini"):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You're an expert with sharp insights. Your job is to identify hidden context or scarce information in a given text to generate clear and concise questions.

For example, when a user asks “how to set environment variables,” you might want to know what operating system they are using (Windows, macOS, Linux, etc.).

You can create one or multiple questions. The answer should consolidate all of the generated questions into a single sentence that flows naturally.

Example: If questions like “are you using windows OS or mac OS” and “do you need examples or not” are generated, the answer should be something like “are you using Windows OS or macOS, and do you need examples of setting environment variables?”.
         
When outputting messages, output a language that matches the user's language.
For example, if the user entered Korean, output the response in Korean """),
        ("user", """Please print the question for the following text:
--- TEXT START ---
{text}
--- TEXT END ---
""")
    ])
    
    model = ChatOpenAI(
        model=model_name,
        temperature=0.6
    )
    
    chain = prompt | model | StrOutputParser()
    
    try:
        response = await chain.ainvoke({"text": text})
        return response
        
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}
    
async def generate_final_prompt(original_english_prompt: str, enhancement_results: Dict[str, str], model_name: str) -> AsyncGenerator[str, None]:
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
    {enhancements_text}

    If the Improvement Suggestions section has a “사용자 추가 피드백 (HITL)” item, please prioritize this item.

    For example, if the original prompt said “I'm using a Windows OS”, but the user responded “I'm actually using a Mac OS” in the “사용자 추가 피드백 (HITL)” section, we'd like to proceed with “Mac OS, not Windows OS”.

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

    async for korean_chunk in chain.astream({
        "original_english_prompt": original_english_prompt,
        "enhancements_text": enhancements_text
    }):
        if korean_chunk:
            yield korean_chunk