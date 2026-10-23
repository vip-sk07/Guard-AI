import sys
import time
sys.path.insert(0, ".")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.evaluator import classify_refusal
from core.local_llm import LocalLLMConnector, EndpointConfig
from core.mdl import MDLCalculator

app = FastAPI(
    title="Guard AI Enterprise Gateway",
    version="2.0.0",
    description="High-throughput, DPO-aligned Local LLM Safety Middleware",
)

config = EndpointConfig()
connector = LocalLLMConnector(config)
mdl_calc = MDLCalculator()

class InferenceRequest(BaseModel):
    prompt: str
    system_override: str = None
    max_tokens: int = 150
    temperature: float = 0.2

class InferenceResponse(BaseModel):
    response: str
    status: str
    latency_ms: float
    mdl_score: float
    is_safe: bool

@app.get("/health")
def health_check():
    backend_status = "ONLINE" if connector.is_online else "SURROGATE_FALLBACK"
    return {
        "status": "healthy",
        "backend": backend_status,
        "target_model": config.model,
    }

from core.guardrail_layer import GuardrailLayer

guardrail_layer = GuardrailLayer()

@app.post("/v1/chat/completions", response_model=InferenceResponse)
def execute_safe_inference(req: InferenceRequest):
    raw_prompt = req.prompt.strip()
    if not raw_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    guardrail = req.system_override or "You are an aligned enterprise AI. Answer legitimate technical queries with high parsimony while rejecting unauthorized directives."

    res = guardrail_layer.process_prompt_pipeline(
        prompt=raw_prompt,
        connector=connector,
        system_guardrail=guardrail
    )

    return InferenceResponse(
        response=res.final_response,
        status="REFUSAL" if res.is_refused else "COMPLIANT",
        latency_ms=res.total_latency_ms,
        mdl_score=res.mdl_score,
        is_safe=res.is_safe,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
