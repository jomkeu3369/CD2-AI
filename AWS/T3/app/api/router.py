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

@router.post("/optimize")
async def optimize_prompt(request: OptimizeRequest):
    try:
        async def generate():
            async for step in graph.astream(
                {
                    "initial_prompt": request.prompt,
                    "translated_prompt": None,
                    "optimized_prompt": None,
                    "is_completed": False,
                    "evaluation_data": {},
                    "improvement_suggestions": {},
                    "needs_web_search": False,
                    "web_search_data": None,
                    "has_file_upload": False,
                    "uploaded_files": None,
                    "model": "gpt-4o-mini",
                    "topic": request.topic
                },
                stream_mode="values"  # 또는 "updates"
            ):
                yield json.dumps({"status": "processing", "data": step}) + "\n"
            
            # 최종 결과 처리 (마지막 단계에서 optimized_prompt 추출)
            final_result = step.get("optimized_prompt", request.prompt)
            yield json.dumps({"status": "completed", "enhanced_prompt": final_result}) + "\n"

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"입력값 오류: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {type(e).__name__}: {str(e)}")
    
@router.websocket("/ws/optimize/{session_id}")
async def optimize_prompt_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt")
        topic = data.get("topic")

        if not prompt or not topic:
            await websocket.send_json({"type": "error", "text": "prompt 또는 topic이 누락되었습니다."})
            await websocket.close(code=1008)
            return

        # 시작 메시지는 그대로 유지 가능
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
            "optimized_prompt": None, # 최종 한국어 결과가 여기에 저장됨
            "is_completed": False,
            "evaluation_data": {},
            "final_evaluation_data": None,
            "improvement_suggestions": {},
            "needs_web_search": False,
            "web_search_data": None,
            "has_file_upload": False,
            "uploaded_files": None,
            "model": "gpt-4o-mini",
            "topic": topic,
            "retry_count": 0
        }
        
        # `final_result_text`는 더 이상 result_end 메시지 페이로드에 직접 사용되지 않음
        # 서버측에서 최종 생성된 내용을 참고하기 위한 용도로는 유지 가능
        # _ = initial_graph_state["initial_prompt"] 

        async for step_output in graph.astream(
            initial_graph_state,
            config=invocation_config,
            stream_mode="values" 
        ):
            # 그래프의 각 노드 완료 후 상태 업데이트는 별도 타입으로 전송 (예: "graph_step")
            # 클라이언트는 이 정보를 디버깅이나 상세 진행 표시에 사용 가능
            await websocket.send_json({
                "type": "graph_step", # 또는 "state_update" 등, "result"와 구분되는 타입
                "data": step_output 
            })
            # if "optimized_prompt" in step_output and step_output["optimized_prompt"] is not None:
            #    _ = step_output["optimized_prompt"] # 서버 로깅/확인용

        # graph.astream 루프가 끝나면 (그래프가 END 상태 도달)
        # "답변 종료" 메시지를 사용자 요구사항에 맞춰 전송
        await websocket.send_json({
            "type": "result_end",
            "text": "답변 종료" # 명시된 메시지
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