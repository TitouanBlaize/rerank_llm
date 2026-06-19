"""
Main FastAPI application for the Mistral document reranker.
"""
import os
from mistralai import Mistral
from fastapi import FastAPI

app = FastAPI()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/chat/{question}")
def answer_question(question: str):
    response = client.chat.complete(
        model="mistral-medium-3-5",
        messages=[{"role": "user", "content": question}],
    )

    return {"response": response.choices[0].message.content}
