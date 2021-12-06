import uvicorn
from fastapi import FastAPI

from emo_classifier import load_classifier
from emo_classifier.api import Comment, Prediction

classifier = load_classifier()
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/thresholds")
async def thresholds():
    return classifier.thresholds.as_dict()


@app.post("/prediction")
async def prediction(comment: Comment) -> Prediction:
    return classifier.predict(comment)


def start():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
