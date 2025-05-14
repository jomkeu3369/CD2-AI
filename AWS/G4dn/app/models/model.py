import os
from dotenv import load_dotenv

from fastapi import WebSocket

from langchain_teddynote import logging
from langgraph.graph import StateGraph, START, END
from langchain_teddynote.graphs import visualize_graph

from app.models.schema import MainState
from app.models.nodes import *

from typing import Dict

load_dotenv()
logging.langsmith("deploy_models")
DB_PATH = os.getenv("DB_path")

workflow = StateGraph(MainState)

workflow.add_node("analyze_prompt", analyze_prompt)
workflow.add_node("analyze", analyze_evaluation_data)
workflow.add_node("suggest", generate_improvement_suggestions)
workflow.add_node("enhance", enhance_prompt)

workflow.set_entry_point("analyze_prompt")
workflow.add_edge("analyze_prompt", "analyze")
workflow.add_edge("analyze", "suggest")
workflow.add_edge("suggest", "enhance")
workflow.add_edge("enhance", END)

graph = workflow.compile()


