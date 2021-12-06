from pydantic import BaseModel


class Comment(BaseModel):
    id: str
    text: str


class Prediction(BaseModel):
    id: str
    labels: list[str]
