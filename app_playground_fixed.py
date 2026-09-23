import os
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langserve import add_routes

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it in Render Environment Variables.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=API_KEY,
    temperature=0.2,
)

class CrewState(TypedDict, total=False):
    task: str
    code: str
    tests: str
    report: str

def response_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)

def clean_code(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

def developer_node(state: CrewState) -> CrewState:
    task = state["task"]
    prompt = f"""
You are a senior digital design engineer.

Generate synthesizable Verilog HDL for this hardware requirement:

{task}

Requirements:
1. Use Verilog-2001 unless SystemVerilog is explicitly required.
2. Generate complete compilable Verilog.
3. Include a complete module declaration and all required ports.
4. Use clear signal names and useful comments.
5. Do not invent unnecessary hardware.
6. Handle reset behavior exactly as requested.
7. Return ONLY Verilog source code.
8. Do NOT use Markdown code fences.
"""
    response = llm.invoke(prompt)
    return {**state, "code": clean_code(response_to_text(response))}

def tester_node(state: CrewState) -> CrewState:
    task = state["task"]
    code = state["code"]
    prompt = f"""
You are a senior Verilog verification engineer.

Hardware requirement:
{task}

Generated Verilog:
{code}

Create a self-contained Verilog testbench.

Requirements:
1. Instantiate the generated module.
2. Generate required clocks.
3. Apply reset correctly.
4. Test normal operation and important boundary cases.
5. Display useful simulation results.
6. Finish with $finish.
7. Return ONLY Verilog testbench source code.
8. Do NOT use Markdown code fences.
"""
    response = llm.invoke(prompt)
    return {
        **state,
        "tests": clean_code(response_to_text(response)),
        "report": "Verilog design and verification testbench generated successfully.",
    }

workflow = StateGraph(CrewState)
workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)
workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", END)
graph = workflow.compile()

class VerilogInput(BaseModel):
    task: str = Field(
        ...,
        description="Describe the digital hardware you want to generate. Example: Design a 4-bit synchronous up counter with reset.",
    )

class VerilogOutput(BaseModel):
    task: str
    code: str
    tests: str
    report: str

def run_verilog_agent(input_data: Any) -> Dict[str, Any]:
    # LangServe can pass the JSON request as a dict. Accept both a
    # Pydantic model and a plain dict so the playground reliably works.
    if isinstance(input_data, VerilogInput):
        task = input_data.task
    elif isinstance(input_data, dict):
        task = input_data.get("task", "")
    else:
        task = getattr(input_data, "task", "")

    if not task or not str(task).strip():
        raise ValueError("Please provide a Verilog design task in the 'task' field.")

    result = graph.invoke({"task": str(task).strip()})
    return {
        "task": result.get("task", str(task).strip()),
        "code": result.get("code", ""),
        "tests": result.get("tests", ""),
        "report": result.get("report", ""),
    }

verilog_agent = RunnableLambda(run_verilog_agent).with_types(
    input_type=VerilogInput,
    output_type=VerilogOutput,
)

app = FastAPI(
    title="Verilog AI Agent",
    version="1.0.0",
    description="AI-powered Verilog and testbench generator using LangGraph and Gemini.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

add_routes(
    app,
    verilog_agent,
    path="/agent",
    playground_type="default",
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "Verilog AI Agent", "playground": "/agent/playground/"}

@app.get("/")
def root():
    return {
        "message": "Verilog AI Agent is running",
        "playground": "/agent/playground/",
        "docs": "/docs",
        "health": "/health",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
