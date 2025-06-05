from pydantic import BaseModel

model_list = [
        {
            "id": 0,
            "name" : "gpt-4o-mini",
            "type": "api"
        },
        {
            "id": 1,
            "name" : "gpt-4o",
            "type": "api"
        },
        {
            "id": 2,
            "name" : "gpt-4.1",
            "type": "api"
        },
        {
            "id": 3,
            "name" : "gpt-4.1-mini",
            "type": "api"
        }
    ]

class ReinforceRequest(BaseModel):
    token: str
    message_id: str
    recommand: bool