# categorization_service/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from categorization import categorize_text

app = FastAPI()

class TranscriptionRequest(BaseModel):
    transcription: str

@app.get("/")
async def root():
    return {"message": "Categorization Service is running"}

@app.post("/categorize")
async def categorize_text_request(transcription: TranscriptionRequest):
    try:
        category = categorize_text(transcription.transcription)
        return {"category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
