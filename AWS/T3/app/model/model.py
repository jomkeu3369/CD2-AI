from langgraph.graph import StateGraph, START, END
from app.model.nodes import *

from langchain_teddynote.graphs import visualize_graph

def should_retry_or_stream(state: MainState) -> str:
    evaluation_score = state.get("final_evaluation_data", {}).get("average_score", 0.0)
    retry_count = state.get("retry_count", 0)
    max_retries = 1

    if evaluation_score >= 0.6:
        return "stream_answer"
    
    elif retry_count < max_retries:
        return "retry_improvement"
    
    else:
        state["optimized_prompt"] = "최적의 답변을 생성하지 못했습니다. 다시 시도해주세요."
        return "stream_answer"

workflow = StateGraph(MainState)
workflow.add_node("topic_evaluation", topic_evaluation_node)
workflow.add_node("translation_prompt", translation_prompt_node)
workflow.add_node("evaluate_prompt", evaluate_prompt_node)
workflow.add_node("improvement_prompt", improvement_prompt_node)
workflow.add_node("enhance_prompt", enhance_prompt_node)
workflow.add_node("evaluate_enhance_prompt", evaluate_enhance_prompt_node)
workflow.add_node("stream_final_answer", stream_final_answer_node) 
workflow.add_node("prepare_for_retry", prepare_for_retry_node)


workflow.add_edge(START, "topic_evaluation")
workflow.add_edge("topic_evaluation", "translation_prompt")
workflow.add_edge("translation_prompt", "evaluate_prompt")
workflow.add_edge("evaluate_prompt", "improvement_prompt")
workflow.add_edge("improvement_prompt", "enhance_prompt")
workflow.add_edge("enhance_prompt", "evaluate_enhance_prompt")

workflow.add_conditional_edges(
    "evaluate_enhance_prompt",
    should_retry_or_stream,
    {
        "stream_answer": "stream_final_answer",
        "retry_improvement": "prepare_for_retry"
    }
)
workflow.add_edge("prepare_for_retry", "improvement_prompt")
workflow.add_edge("stream_final_answer", END)

graph = workflow.compile()