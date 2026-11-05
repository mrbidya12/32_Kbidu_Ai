# serving.py
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import asyncio
from starlette.responses import StreamingResponse
import json
import time

from model import KbiduForCausalLM
from tokenizer import Tokenizer
from inference import Inference
from config import Kbidu30BConfig

app = FastAPI(title="Kbidu AI", description="30B Causal Language Model")

# Global variables
model = None
tokenizer = None
inference_engine = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    use_cot: Optional[bool] = True
    stream: Optional[bool] = False

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    time_taken: float

@app.on_event("startup")
def load_model():
    global model, tokenizer, inference_engine
    config = Kbidu30BConfig()
    tokenizer = Tokenizer("data/tokenizer/tokenizer.json")
    model = KbiduForCausalLM(config)
    # Load checkpoint if available
    import os
    if os.path.exists("checkpoints/model_step_100000.pt"):
        model.load_state_dict(torch.load("checkpoints/model_step_100000.pt", map_location=device))
    model.to(device)
    model.eval()
    inference_engine = Inference(model, tokenizer, device, max_new_tokens=200)
    print("✅ Kbidu AI loaded successfully")

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    start_time = time.time()
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if request.stream:
        # Streaming response
        async def stream_generator():
            yield "data: " + json.dumps({"status": "start"}) + "\n\n"
            # For streaming, we generate incrementally (simplified)
            input_ids = torch.tensor([tokenizer.encode(request.prompt)]).to(device)
            for _ in range(request.max_new_tokens):
                with torch.no_grad():
                    logits = model(input_ids)
                    next_logits = logits[0, -1, :] / request.temperature
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(device)], dim=1)
                    if next_token == tokenizer.eos_id:
                        break
                    token_text = tokenizer.decode([next_token])
                    yield "data: " + json.dumps({"token": token_text}) + "\n\n"
            yield "data: " + json.dumps({"status": "done"}) + "\n\n"
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        # Non-streaming
        text = inference_engine.generate(
            request.prompt, 
            use_cot=request.use_cot, 
            temperature=request.temperature
        )
        return GenerateResponse(
            text=text,
            tokens_generated=len(text.split()),
            time_taken=time.time() - start_time
        )

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "Kbidu-30B"}

# Run with: uvicorn serving:app --host 0.0.0.0 --port 8000