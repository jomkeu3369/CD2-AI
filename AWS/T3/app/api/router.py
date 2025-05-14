import os
import httpx

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import json
from app.api.schema import OptimizeResponse, OptimizeRequest
from app.model.model import graph

router = APIRouter(
    prefix="/api/v1"
)

@router.get("/")
def read_root():
    return {"message": "W.A.F.Q. AI SERVER is running!"}

@router.websocket("/ws/optimize/{session_id}")
async def optimize_prompt_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    try:
        data:dict = await websocket.receive_json()
        token = data.get("token")
        prompt = data.get("prompt")
        topic = data.get("topic")
        option:dict = data.get("option")
        option_websearch = option.get("web_search")
        option_file = option.get("file")
        option_optimize = option.get("optimize")
        option_model = option.get("model")

        if not prompt or not topic:
            await websocket.send_json({"type": "error", "text": "prompt 또는 topic이 누락되었습니다."})
            await websocket.close(code=1008)
            return

        await websocket.send_json({"type": "result_start", "text": "답변 생성을 시작합니다."})
        
        invocation_config = {
            "configurable": {
                "websocket": websocket,
                "session_id": session_id
            }
        }

        initial_graph_state = {
            "initial_prompt": prompt,
            "translated_prompt": None,
            "optimized_prompt": None,
            "is_completed": False,
            "evaluation_data": {},
            "final_evaluation_data": None,
            "improvement_suggestions": {},
            "needs_web_search": False,
            "web_search_data": None,
            "has_file_upload": False,
            "uploaded_files": None,
            "model": option_model,
            "topic": topic,
            "retry_count": 0
        }
    
        async for step_output in graph.astream(
            initial_graph_state,
            config=invocation_config,
            stream_mode="values" 
        ):
            await websocket.send_json({
                "type": "cot",
                "data": step_output 
            })

        await websocket.send_json({
            "type": "result_end",
            "text": "답변 종료"
        })

    except WebSocketDisconnect:
        print(f"Session {session_id} disconnected.")
    except json.JSONDecodeError:
        print(f"Session {session_id}: Invalid JSON received.")
        await websocket.send_json({"type": "error", "text": "잘못된 JSON 형식입니다."})
    except Exception as e:
        error_message = f"서버 내부 오류 발생: {type(e).__name__}: {str(e)}"
        print(f"Session {session_id}: {error_message}")
        try:
            await websocket.send_json({
                "type": "error",
                "text": error_message
            })
        except Exception:
            pass