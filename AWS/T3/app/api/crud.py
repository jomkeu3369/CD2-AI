from fastapi import WebSocket
from typing import Dict, Optional

from langgraph.graph import StateGraph
from app.models.schema import MainState
from app.models.model import graph

class LangGraphManager:
    def __init__(self):
        self.workflows: Dict[WebSocket, StateGraph] = {}
        self.states: Dict[WebSocket, MainState] = {}

    async def init_workflow(self, websocket: WebSocket):
        self.workflows[websocket] = graph
        self.states[websocket] = MainState()

    async def process_node(self, state: MainState):
        state["processed"] = True
        return state

    async def human_check_node(self, state: MainState):
        state.requires_human = True
        return state
    
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, bool] = {}
        self.lang_graph = LangGraphManager()

    async def handle_message(self, websocket: WebSocket, data: str):
        current_state = self.lang_graph.states[websocket]
        
        if current_state.requires_human:
            current_state.pending_input = data
            current_state.requires_human = False
            await self.process_workflow(websocket)
        else:
            await self.process_workflow(websocket, data)

    async def process_workflow(self, websocket: WebSocket, data: Optional[str] = None):
        workflow = self.lang_graph.workflows[websocket]
        current_state = self.lang_graph.states[websocket]
        
        if data:
            current_state.workflow_state["input"] = data
        
        new_state = await workflow.arun(current_state)
        
        if new_state.requires_human:
            await websocket.send_text("HUMAN_INPUT_REQUIRED")
        else:
            await websocket.send_text(f"RESULT: {new_state.workflow_state}")

status_manager = ConnectionManager()
