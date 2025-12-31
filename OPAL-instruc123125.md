Here is a comprehensive, detailed instruction prompt designed to be used in OPAL (or any advanced coding agent) to build the application.

You can copy and paste the block below directly into the tool.

***

# OPAL Instruction Prompt

**Role:** You are a Senior AI Solutions Architect and Full-Stack Streamlit Developer.

**Objective:** Create a fully functional, production-ready "FDA 510(k) Agentic AI Review System" based on the provided source code, configuration files, and skill documentation.

**Context:** This application is a multi-agent system designed to assist regulatory affairs professionals in reviewing FDA medical device submissions. It utilizes multiple LLM providers (OpenAI, Google, Anthropic, xAI) and performs tasks ranging from PDF extraction to complex orchestration and risk analysis.

**Instructions:**
Please generate the complete project structure and files necessary to run this application. Follow the step-by-step implementation plan below.

---

### Step 1: Environment & Dependencies (`requirements.txt`)
Create a `requirements.txt` file including all necessary libraries found in the imports of the provided code. Ensure specific versions are pinned for stability where appropriate.
*   **Core:** `streamlit`, `pyyaml`, `pandas`
*   **LLM Providers:** `openai`, `google-generativeai`, `anthropic`, `httpx` (for Grok)
*   **Document Processing:** `pypdf`, `python-docx`

### Step 2: Configuration (`agents.yaml`)
Create a file named `agents.yaml`. You must populate this file **exactly** with the YAML content provided below. This file defines the persona, model, and system prompts for every agent in the system.

**[INSERT content from the provided `agents.yaml` here]**

### Step 3: Application Logic (`app.py`)
Create the main application file `app.py`. Use the provided Python code as the source, but ensure the following strict requirements are met:
1.  **Architecture:** Maintain the exact class structure (`ModelConfig`, `UIConfig`) and the Unified LLM Interface (`call_llm` function) that routes requests to OpenAI, Gemini, Anthropic, or Grok based on the model name.
2.  **Session State:** Ensure robust session state management so data persists between tab switches (e.g., extracted PDF text needs to be available in the "Summary" tab).
3.  **UI/UX:**
    *   Implement the Sidebar with "Painter Style" injection (dynamic CSS).
    *   Implement the Bilingual Dictionary (`LABELS`) for English/Traditional Chinese toggling.
    *   Implement the Tab structure exactly as defined (Dashboard, 510k, PDF, Summary, Diff, Checklist, Notes, Orchestration, Dynamic).
4.  **Error Handling:** Ensure `try/except` blocks surround all API calls and file parsers (PDF/DOCX) to prevent app crashes.
5.  **Dynamic Loading:** The app must load the `agents.yaml` file created in Step 2. If the file is missing, provide a graceful fallback or error message.

**[INSERT content from the provided Python Code here]**

### Step 4: Documentation & Capabilities (`README.md`)
Create a professional `README.md` file. Use the provided `SKILL.md` content to generate a "System Capabilities" section.
*   Summarize the "Skills Demonstrated" (System Architecture, Regulatory Expertise, Document Processing) into a readable format.
*   Include instructions on how to set up API keys in the `.streamlit/secrets.toml` or via the Sidebar UI.
*   Explain how to launch the app (`streamlit run app.py`).

**[INSERT content from the provided `SKILL.md` here]**

---

### Execution Plan
1.  **Analyze** the provided Python code to ensure all imports are covered in `requirements.txt`.
2.  **Write** `agents.yaml` to disk.
3.  **Write** `app.py` to disk.
4.  **Write** `README.md` to disk.
5.  **Verification:** Ensure that the `agent_run_ui` function correctly reads from the loaded YAML configuration to populate the "System Prompt" dynamically.

**Output:** Please generate the code blocks for these four files now.
