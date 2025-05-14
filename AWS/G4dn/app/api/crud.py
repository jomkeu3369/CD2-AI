from fastapi import WebSocket, WebSocketDisconnect # WebSocketDisconnect 임포트
from typing import Dict, Optional
import asyncio # asyncio 임포트 추가

# from langgraph.graph import StateGraph # StateGraph 직접 사용하지 않음
from app.models.schema import MainState
from app.models.model import graph # graph 직접 임포트

class LangGraphManager: # 이 클래스는 현재 상태에서 크게 수정할 필요는 없습니다.
    def __init__(self):
        # 웹소켓별 LangGraph 인스턴스나 상태를 저장할 필요가 있다면 여기에 구현합니다.
        # 현재는 graph를 직접 사용하므로, 이 클래스의 역할이 줄어들 수 있습니다.
        self.workflows: Dict[WebSocket, any] = {} # 필요시 graph 저장
        self.states: Dict[WebSocket, MainState] = {} # 개별 상태 저장 공간

    async def init_workflow(self, websocket: WebSocket):
        # self.workflows[websocket] = graph # graph는 전역이므로 굳이 저장 안해도 됨
        # self.states[websocket] = MainState() # MainState는 process_workflow에서 생성
        print(f"Workflow initialized for {websocket.client}")
        pass


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, bool] = {} # 현재 활성화된 연결 추적용
        self.lang_graph = LangGraphManager() # LangGraphManager 인스턴스

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = True
        # await self.lang_graph.init_workflow(websocket) # init_workflow 호출 시점 변경 고려

    def cleanup_connection(self, websocket: WebSocket):
        if websocket in self.lang_graph.workflows:
            del self.lang_graph.workflows[websocket]
        if websocket in self.lang_graph.states:
            del self.lang_graph.states[websocket]
        if websocket in self.active_connections:
            del self.active_connections[websocket]
        print(f"Cleaned up resources for disconnected websocket: {websocket.client}")

    async def process_workflow(self, websocket: WebSocket, data: Optional[str] = None):
        # init_workflow를 여기서 호출하거나, connect 시점에 호출하도록 변경 가능
        # 매번 새로운 상태로 시작해야 하므로, init_workflow의 역할은 여기서 상태 객체 생성으로 대체될 수 있음.
        # await self.lang_graph.init_workflow(websocket) # 만약 연결당 상태를 유지해야 한다면 connect 시점에 호출

        try:
            initial_state_params = {
                "original_prompt": data,
                "evaluation_data": dict(),
                "weak_categories": [],
                "improvement_suggestions": {},
                "enhanced_prompt": "",
                "execution_log": [],
                "exit": False
            }
            
            # LangGraph 전체 실행
            # graph.ainvoke는 MainState TypedDict를 직접 받지 않고, 딕셔너리를 받습니다.
            final_state_dict = await graph.ainvoke(initial_state_params)

            # TypedDict로 변환 (선택 사항, 그러나 타입 힌팅 및 자동완성에 도움)
            final_state: MainState = final_state_dict # type: ignore 

            enhanced_prompt = final_state.get("enhanced_prompt")
            error_message_from_state = final_state.get("error_message") # nodes.py 에서 error_message 설정 가능
            exit_flag = final_state.get("exit", False)

            if exit_flag or error_message_from_state:
                error_msg = error_message_from_state or "Optimization process was exited or an error occurred."
                log_detail = final_state.get("execution_log", [])
                await websocket.send_json({"type": "error", "message": error_msg, "details": log_detail})
                return

            if enhanced_prompt:
                await websocket.send_json({"type": "stream_start", "message": "Final enhanced prompt streaming started."})
                
                chunk_size = 20
                for i in range(0, len(enhanced_prompt), chunk_size):
                    chunk = enhanced_prompt[i:i + chunk_size]
                    await websocket.send_json({"type": "stream_chunk", "content": chunk})
                    await asyncio.sleep(0.05)

                await websocket.send_json({"type": "stream_end", "message": "Final enhanced prompt streaming finished."})
                
                eval_data = final_state.get('evaluation_data')
                eval_summary = {}
                if eval_data and isinstance(eval_data, dict):
                    for cat, cat_data_val in eval_data.items():
                        if isinstance(cat_data_val, dict):
                            eval_summary[cat] = round(cat_data_val.get('average_score', 0.0), 3)
                
                suggestions = final_state.get('improvement_suggestions')
                sugg_count = len(suggestions) if suggestions else 0

                summary_payload = {
                    "original_prompt": final_state.get('original_prompt', data),
                    "enhanced_prompt_preview": enhanced_prompt[:200] + "..." if enhanced_prompt and len(enhanced_prompt) > 200 else enhanced_prompt,
                    "weak_categories": final_state.get('weak_categories'),
                    "suggestions_count": sugg_count,
                    "evaluation_summary": eval_summary if eval_summary else None,
                }
                await websocket.send_json({"type": "final_summary", "data": summary_payload})

            else:
                await websocket.send_json({"type": "info", "message": "Enhanced prompt is empty or could not be generated."})

        except WebSocketDisconnect:
            print(f"WebSocket disconnected during process_workflow: {websocket.client}")
            self.cleanup_connection(websocket)
        except Exception as e:
            error_detail_msg = f"An error occurred during workflow processing: {type(e).__name__} - {str(e)}"
            print(error_detail_msg)
            try:
                await websocket.send_json({"type": "error", "message": error_detail_msg})
            except Exception as send_err:
                print(f"Failed to send error to client {websocket.client}: {send_err}")
status_manager = ConnectionManager()