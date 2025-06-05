from pydantic import BaseModel

model_list = [
        {
            "id": 0,
            "name" : "gemma3:12b",
            "type": "local"
        },
        {
            "id": 1,
            "name" : "llama3.1:8b",
            "type": "local"
        }
    ]

class ReinforceRequest(BaseModel):
    token: str
    message_id: str
    recommand: bool