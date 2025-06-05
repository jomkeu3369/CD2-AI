import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.model.schema import MainState
from app.log import setup_logging

from app.model.nodes import (
    topic_evaluation_node,
    translation_prompt_node,
    evaluate_prompt_node,
    improvement_prompt_node,
    enhance_prompt_node,
    stream_error_node,
    dr_initializer_node, 
    dr_thought_generator_node,
    dr_evaluator_node,
    dr_tot_control_node,
    dr_markdown_report_generator_node,
    hitl_request_node
)

logger: logging.Logger = setup_logging()

def should_generate_report(state: MainState) -> str:
    """
    Determines whether to proceed to report generation or end the workflow.
    """
    if state.get("is_completed", False) and not state.get("generate_detailed_report", False) :
        return "end_workflow"
    
    if state.get("generate_detailed_report", False):
        logger.debug("Conditional edge: Report generation is requested.")
        return "start_report_generation"
    else:
        logger.debug("Conditional edge: Report generation is not requested. Ending workflow.")
        return "end_workflow"

def dr_should_continue_tot(state: MainState) -> str:
    """
    Determines the next step in the Tree of Thoughts (ToT) process for detailed report generation.
    """
    report_state = state.get("report_data")
    if not report_state:
        logger.warning("Conditional edge (ToT): Report data is missing. Proceeding to generate report (or end).")
        return "generate_report"
        
    if report_state.get('dr_current_depth', 0) >= report_state.get('dr_max_depth', 1):
        logger.debug(f"Conditional edge (ToT): Max depth ({report_state.get('dr_max_depth')}) reached. Proceeding to generate report.")
        return "generate_report"
    
    if not report_state.get('dr_active_thoughts'):
        logger.debug("Conditional edge (ToT): No active thoughts. Proceeding to generate report.")
        return "generate_report"

    logger.debug("Conditional edge (ToT): Conditions met to continue thought generation.")
    return "generate_thoughts"

workflow = StateGraph(MainState)

workflow.add_node("topic_evaluation", topic_evaluation_node)
workflow.add_node("translation_prompt", translation_prompt_node)
workflow.add_node("evaluate_prompt", evaluate_prompt_node)
workflow.add_node("improvement_prompt", improvement_prompt_node)
workflow.add_node("hitl_request", hitl_request_node)
workflow.add_node("enhance_prompt", enhance_prompt_node)
workflow.add_node("stream_error", stream_error_node)

workflow.add_node("dr_initializer", dr_initializer_node)
workflow.add_node("dr_thought_generator", dr_thought_generator_node)
workflow.add_node("dr_evaluator", dr_evaluator_node)
workflow.add_node("dr_tot_control", dr_tot_control_node)
workflow.add_node("dr_markdown_generator", dr_markdown_report_generator_node)

workflow.add_edge(START, "topic_evaluation")

workflow.add_conditional_edges(
    "topic_evaluation",
    lambda state: "STOP" if state.get("is_completed") else "NEXT",
    {
        "STOP": "stream_error", 
        "NEXT": "translation_prompt"
    }
)
workflow.add_edge("stream_error", END)

workflow.add_edge("translation_prompt", "evaluate_prompt")
workflow.add_edge("translation_prompt", "hitl_request")

workflow.add_edge("evaluate_prompt", "improvement_prompt")

workflow.add_edge(["improvement_prompt", "hitl_request"], "enhance_prompt")


workflow.add_conditional_edges(
    "enhance_prompt",
    should_generate_report,
    {
        "start_report_generation": "dr_initializer",
        "end_workflow": END
    }
)

workflow.add_edge("dr_initializer", "dr_tot_control")

workflow.add_conditional_edges(
    "dr_tot_control",
    dr_should_continue_tot,
    {
        "generate_thoughts": "dr_thought_generator",
        "generate_report": "dr_markdown_generator"
    }
)
workflow.add_edge("dr_thought_generator", "dr_evaluator")
workflow.add_edge("dr_evaluator", "dr_tot_control")

workflow.add_edge("dr_markdown_generator", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

logger.info("LangGraph workflow compiled successfully with memory saver.")

