import os
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from langserve import add_routes


# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it in Render -> Environment."
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=API_KEY,
    temperature=0.2,
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class VerilogState(TypedDict, total=False):
    task: str
    code: str
    tests: str
    report: str


# ============================================================
# HELPERS
# ============================================================

def response_to_text(response: Any) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = []

        for item in content:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    result.append(str(item["text"]))
            else:
                result.append(str(item))

        return "\n".join(result)

    return str(content)


def remove_markdown_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines)

    return text.strip()


# ============================================================
# DEVELOPER NODE
# ============================================================

def developer_node(state: VerilogState) -> VerilogState:

    task = state["task"]

    prompt = f"""
You are an expert digital hardware designer.

Generate complete synthesizable Verilog HDL for this requirement:

{task}

Rules:
- Use Verilog-2001 syntax unless SystemVerilog is explicitly requested.
- Generate a complete module.
- Include all required inputs and outputs.
- Use correct synthesizable logic.
- Use meaningful signal names.
- Add short useful comments.
- Follow the user's requested behavior exactly.
- Do not add unnecessary hardware.
- Return ONLY Verilog source code.
- Do NOT use Markdown code fences.
"""

    response = llm.invoke(prompt)

    code = remove_markdown_fences(response_to_text(response))

    return {
        "task": task,
        "code": code,
    }


# ============================================================
# TESTER NODE
# ============================================================

def tester_node(state: VerilogState) -> VerilogState:

    task = state["task"]
    code = state["code"]

    prompt = f"""
You are an expert Verilog verification engineer.

Hardware requirement:
{task}

Generated Verilog:
{code}

Generate a complete Verilog testbench.

Rules:
- Instantiate the generated DUT.
- Generate required clock signals if needed.
- Apply reset if required.
- Test all important input combinations.
- Test normal operation.
- Test important boundary cases.
- Display useful simulation information.
- End simulation using $finish.
- Return ONLY Verilog testbench source code.
- Do NOT use Markdown code fences.
"""

    response = llm.invoke(prompt)

    tests = remove_markdown_fences(response_to_text(response))

    return {
        **state,
        "tests": tests,
        "report": "Verilog design and verification testbench generated successfully.",
    }


# ============================================================
# LANGGRAPH
# ============================================================

workflow = StateGraph(VerilogState)

workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", END)

graph = workflow.compile()


# ============================================================
# LANGSERVE API SCHEMA
# ============================================================

class VerilogInput(BaseModel):
    task: str = Field(
        ...,
        description="Describe the hardware you want to generate."
    )


class VerilogOutput(BaseModel):
    task: str
    code: str
    tests: str
    report: str


def run_agent(data: Any) -> Dict[str, str]:

    # LangServe may provide either a Pydantic object or a dictionary.
    if isinstance(data, VerilogInput):
        task = data.task
    elif isinstance(data, dict):
        task = data.get("task", "")
    else:
        task = getattr(data, "task", "")

    if not task or not task.strip():
        raise ValueError("Please enter a hardware design requirement.")

    result = graph.invoke({
        "task": task.strip()
    })

    return {
        "task": result.get("task", task),
        "code": result.get("code", ""),
        "tests": result.get("tests", ""),
        "report": result.get(
            "report",
            "Generation completed."
        ),
    }


verilog_agent = RunnableLambda(run_agent).with_types(
    input_type=VerilogInput,
    output_type=VerilogOutput,
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Verilog AI Agent",
    version="1.0.0",
    description="AI-powered Verilog code and testbench generator.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LANGSERVE API
# ============================================================

add_routes(
    app,
    verilog_agent,
    path="/agent",
    playground_type="default",
)


# ============================================================
# CUSTOM PLAYGROUND UI
# ============================================================

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Verilog AI Agent</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
}

.header {
    padding: 22px 35px;
    background: #111827;
    border-bottom: 1px solid #334155;
}

.header h1 {
    margin: 0;
    font-size: 25px;
}

.header p {
    margin: 7px 0 0;
    color: #94a3b8;
}

.container {
    max-width: 1400px;
    margin: auto;
    padding: 25px;
}

.input-card {
    background: #1e293b;
    padding: 22px;
    border-radius: 12px;
    margin-bottom: 22px;
}

label {
    display: block;
    margin-bottom: 10px;
    font-weight: bold;
}

textarea {
    width: 100%;
    min-height: 120px;
    resize: vertical;
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 14px;
    background: #0f172a;
    color: #e2e8f0;
    font-size: 15px;
    outline: none;
}

textarea:focus {
    border-color: #60a5fa;
}

button {
    border: none;
    border-radius: 7px;
    padding: 11px 18px;
    cursor: pointer;
    font-weight: bold;
}

.generate {
    margin-top: 14px;
    background: #2563eb;
    color: white;
}

.generate:hover {
    background: #1d4ed8;
}

.generate:disabled {
    background: #475569;
    cursor: not-allowed;
}

.status {
    margin-top: 12px;
    color: #94a3b8;
}

.grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 22px;
}

.card {
    background: #1e293b;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #334155;
}

.card-header {
    padding: 15px 18px;
    background: #111827;
    border-bottom: 1px solid #334155;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.card-header h2 {
    font-size: 17px;
    margin: 0;
}

.actions button {
    background: #334155;
    color: #e2e8f0;
    margin-left: 5px;
    padding: 7px 11px;
    font-size: 12px;
}

pre {
    margin: 0;
    padding: 20px;
    min-height: 300px;
    max-height: 600px;
    overflow: auto;
    white-space: pre-wrap;
    font-family: Consolas, "Courier New", monospace;
    font-size: 14px;
    line-height: 1.55;
    color: #dbeafe;
    background: #020617;
}

.report {
    margin-top: 22px;
}

.report-content {
    padding: 18px;
    color: #86efac;
}

@media (max-width: 900px) {

    .grid {
        grid-template-columns: 1fr;
    }

    .container {
        padding: 15px;
    }

}

</style>

</head>

<body>

<div class="header">

    <h1>⚡ Verilog AI Agent</h1>

    <p>
        Generate Verilog HDL and an automatic verification testbench
        using Gemini + LangGraph.
    </p>

</div>


<div class="container">

    <div class="input-card">

        <label for="task">
            Describe your hardware
        </label>

        <textarea
            id="task"
            placeholder="Example: Generate Verilog code for a 4-bit synchronous up counter with active-high reset."
        ></textarea>

        <button
            id="generateBtn"
            class="generate"
            onclick="generateVerilog()"
        >
            Generate Verilog
        </button>

        <div id="status" class="status"></div>

    </div>


    <div class="grid">

        <div class="card">

            <div class="card-header">

                <h2>Verilog Code</h2>

                <div class="actions">

                    <button onclick="copyText('code')">
                        Copy
                    </button>

                    <button onclick="downloadText('code', 'design.v')">
                        Download
                    </button>

                </div>

            </div>

            <pre id="code">Generated Verilog will appear here...</pre>

        </div>


        <div class="card">

            <div class="card-header">

                <h2>Testbench</h2>

                <div class="actions">

                    <button onclick="copyText('tests')">
                        Copy
                    </button>

                    <button onclick="downloadText('tests', 'testbench.v')">
                        Download
                    </button>

                </div>

            </div>

            <pre id="tests">Generated testbench will appear here...</pre>

        </div>

    </div>


    <div class="card report">

        <div class="card-header">

            <h2>Agent Report</h2>

        </div>

        <div
            id="report"
            class="report-content"
        >
            Waiting for generation...
        </div>

    </div>

</div>


<script>

async function generateVerilog() {

    const task =
        document.getElementById("task").value.trim();

    const button =
        document.getElementById("generateBtn");

    const status =
        document.getElementById("status");

    if (!task) {

        status.textContent =
            "Please describe the hardware you want to generate.";

        return;
    }

    button.disabled = true;

    button.textContent = "Generating...";

    status.textContent =
        "AI agent is generating Verilog and testbench...";

    document.getElementById("code").textContent = "";

    document.getElementById("tests").textContent = "";

    document.getElementById("report").textContent =
        "Processing...";

    try {

        const response = await fetch(
            "/agent/invoke",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    input: {
                        task: task
                    }
                })
            }
        );

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail ||
                data.error ||
                "Agent request failed."
            );

        }

        const output = data.output || data;

        document.getElementById("code").textContent =
            output.code || "No Verilog code returned.";

        document.getElementById("tests").textContent =
            output.tests || "No testbench returned.";

        document.getElementById("report").textContent =
            output.report || "Generation completed.";

        status.textContent =
            "✓ Generation completed successfully.";

    } catch (error) {

        console.error(error);

        status.textContent =
            "❌ Error: " + error.message;

        document.getElementById("report").textContent =
            "Generation failed. Check the Render logs for details.";

    } finally {

        button.disabled = false;

        button.textContent =
            "Generate Verilog";

    }

}


function copyText(elementId) {

    const text =
        document.getElementById(elementId).textContent;

    navigator.clipboard.writeText(text);

}


function downloadText(elementId, filename) {

    const text =
        document.getElementById(elementId).textContent;

    const blob =
        new Blob(
            [text],
            { type: "text/plain" }
        );

    const url =
        URL.createObjectURL(blob);

    const link =
        document.createElement("a");

    link.href = url;

    link.download = filename;

    link.click();

    URL.revokeObjectURL(url);

}

</script>

</body>

</html>
"""


@app.get("/", response_class=HTMLResponse)
async def homepage():

    return HTML_PAGE


@app.get("/health")
async def health():

    return {
        "status": "ok",
        "service": "Verilog AI Agent"
    }


# ============================================================
# START LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port
    )
