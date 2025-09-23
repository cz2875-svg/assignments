#APAN5560 Assignment 1 Part 1
#Chunzi Zhang

from fastapi import FastAPI, Query
from pydantic import BaseModel
import spacy

nlp = spacy.load("en_core_web_lg")

app = FastAPI()

class WordInput(BaseModel):
    word: str

@app.get("/embedding")
def get_embedding(word: str = Query(..., description="query word to embed")):
    doc = nlp(word)
    return {
        "word": word,
        "embedding": doc.vector.tolist(),
        "dim": len(doc.vector)
    }

@app.post("/embedding")
def get_embedding_post(input_data: WordInput):
    doc = nlp(input_data.word)
    return {
        "word": input_data.word,
        "embedding": doc.vector.tolist(),
        "dim": len(doc.vector)
    }

@app.get("/")
def root():
    return {"message": "API with spacy embeddings functionality is implemented. Use /embedding?word=your_word to get the embedding of a word."}
