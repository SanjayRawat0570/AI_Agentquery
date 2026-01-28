from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI(title="Mock LLM")

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    system: str | None = None

@app.post('/api/generate')
async def generate(req: GenerateRequest):
    # Very small mock: echo back a friendly completion based on prompt
    prompt = req.prompt or ''
    # Truncate prompt for brevity
    summary = prompt.strip()[:500]
    response_text = f"[Mocked LLM response] Based on your prompt: {summary}"
    return {"response": response_text}

@app.head('/api/generate')
async def head_generate():
    return {}
