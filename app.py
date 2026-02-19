from fastapi import FastAPI
from pydantic import BaseModel
from agent_core import load_llm, create_tools, create_grok_agent
import os

app = FastAPI(title="Grok-Like Indic Agent")

# Change this path after downloading / fine-tuning
MODEL_PATH = "./models/llama-3.1-8b.Q5_K_M.gguf"

llm = load_llm(MODEL_PATH)
tools = create_tools()
agent_executor = create_grok_agent(llm, tools)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = agent_executor.invoke({"input": request.message})
        return {"response": response["output"]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
