# ResearchAgent: AI Assistant for Research Workflows

This project provides a functional user interface and backend for an AI agent system designed to assist with research tasks, particularly in fields like bioinformatics and epidemiology. It features a three-panel layout (Tasks, Chat, Monitor/Artifact Viewer) and connects via WebSockets to a Python backend powered by **LangChain**. The agent can use configurable LLMs (Google Gemini or local Ollama) for reasoning and various tools (Shell, Web Search, Web Reader, File Read/Write, Package Installer, Python REPL, PubMed Search) to perform actions within isolated task workspaces.

## Features

* **Task Management:** Create, select, delete, and rename tasks. Each task maintains its own context and workspace.
* **Chat Interface:** Interact with the AI agent via a familiar chat window. Supports input history (Up/Down arrows). Basic Markdown rendering (newlines, bold, italics, code blocks, links). A "Thinking..." status line appears while the agent processes requests.
* **LLM Selection:** Choose the specific language model (Gemini or Ollama models configured in `.env`) to use for the current session directly from the chat header. The UI syncs with the backend's actual default LLM on new connections.
* **Agent Workspace (Monitor):** View the agent's internal steps, tool usage, and outputs in a structured, styled log panel.
* **Monitor Status Indicator:** A visual indicator (dot + text) in the monitor header shows the agent's current state (e.g., Idle, Running, Error, Disconnected).
* **Agent Cancellation (STOP Button):** A STOP button appears in the monitor header while the agent is running. Clicking it sends a cancellation request to the backend.
* **File Upload:** An "Upload File(s)" button in the task panel allows users to upload files directly into the currently selected task's workspace. The artifact viewer updates automatically after a successful upload.
* **Artifact Viewer:** Displays generated `.png` images and previews common text files (`.txt`, `.py`, `.csv`, etc.) in a dedicated area below the monitor logs, with navigation for multiple artifacts. The filename is displayed above the content, and the navigation buttons remain fixed at the bottom.
* **LLM Token Tracking:** Displays token usage (input, output, total) for the last LLM call and a running total for the current task in the left panel.
* **Tool Integration:** Includes tools for:
    * Web Search (`duckduckgo_search`)
    * Web Page Reading (`web_page_reader`)
    * PubMed Search (`pubmed_search`)
    * File Reading (`read_file` within task workspace - **Supports text and PDF files**. Reads full PDF content but adds a warning if it exceeds a configurable length.)
    * File Writing (`write_file` within task workspace)
    * Shell Command Execution (`workspace_shell` within task workspace, including `Rscript` if R is installed)
    * Python Package Installation (`python_package_installer`) **(Security Warning!)**
    * Python Code Execution (`Python_REPL`) **(Security Warning!)**
* **Backend:** Python backend using `websockets`, `aiohttp` (for file serving), and `LangChain`.
* **Frontend:** Simple HTML, CSS, and vanilla JavaScript.
* **Configuration:** Extensive configuration via `.env` file (API keys, available models, agent tuning, tool settings, server options).
* **Persistence:** Task list (including names) and chat/monitor history are stored locally using SQLite in the `database/` directory.
* **Task Workspaces:** File/Shell tools operate within isolated directories for each task (`workspace/<task_id>/`), created upon task selection.

## Tech Stack

* **Frontend:** HTML5, CSS3, Vanilla JavaScript (ES6+)
* **Backend:**
    * Python 3.10+ (`asyncio`, `websockets`)
    * **Web Server:** `aiohttp`, `aiohttp-cors`
    * **LangChain Core:** `langchain`
    * **LLM Integrations:** `langchain-google-genai`, `langchain-ollama`
    * **Tools:** `langchain-community` (Search), `langchain-experimental` (Python REPL), `biopython` (PubMed)
    * **Prompts:** Local fallback prompt used (based on `react-chat`)
    * **Config:** `python-dotenv`
    * **HTTP:** `httpx`
    * **Web Parsing:** `beautifulsoup4`, `lxml`
    * **Async File I/O:** `aiofiles`
    * **PDF Reading:** `pypdf`
    * **Plotting (Example):** `matplotlib`
    * **Database:** `aiosqlite`
* **Environment:** `venv` with `pip` (or `uv`)
* **Protocol:** WebSockets (WS), HTTP (for file upload/serving)

## Project Structure



```

ResearchAgent/
├── .venv/              # Virtual environment
├── backend/
│   ├── init.py
│   ├── agent.py        # Agent creation logic
│   ├── callbacks.py    # WebSocket callback handler
│   ├── config.py       # Configuration loading
│   ├── db_utils.py     # SQLite database functions
│   ├── llm_setup.py    # LLM initialization (Gemini/Ollama)
│   ├── server.py       # Main WebSocket & File server logic
│   └── tools.py        # Tool definitions and factory
├── css/
│   └── style.css       # Frontend styling
├── database/           # SQLite database storage (Created automatically, GITIGNORED)
│   └── agent_history.db
├── js/
│   └── script.js       # Frontend JavaScript logic
├── workspace/          # Base directory for task workspaces (GITIGNORED)
│   └── <task_id>/      # Auto-created workspace for each task
│       └── ...         # Files created by the agent for this task
├── .env                # Environment variables (GITIGNORED)
├── .env.example        # Example environment file
├── .gitignore          # Git ignore rules
├── Dockerfile          # Docker build instructions
├── docker-compose.yml  # Docker compose configuration
├── index.html          # Main HTML file for the UI
├── requirements.txt    # Python dependencies
└── README.md           # This file

```

## Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone [https://github.com/vbianchi/ResearchAgent.git](https://github.com/vbianchi/ResearchAgent.git)
    cd ResearchAgent
    ```
2.  **Prerequisites:**
    * Ensure Python 3.10+ is installed.
    * **(Optional):** Install R and ensure `Rscript` is in PATH for R script execution via the shell tool.

3.  **Install `uv` (Recommended - Fast Package Installer):**
    * Follow instructions: [https://github.com/astral-sh/uv#installation](https://github.com/astral-sh/uv#installation)

4.  **Create and Activate Virtual Environment:**
    ```bash
    # Using uv (recommended)
    uv venv --python 3.12 # Or your desired Python version

    # OR using standard venv
    # python -m venv .venv

    # Activate (Linux/Mac/WSL)
    source .venv/bin/activate
    # (Windows CMD: .venv\Scripts\activate.bat)
    # (Windows PowerShell: .venv\Scripts\Activate.ps1)
    ```

5.  **Install Dependencies:**
    ```bash
    # Using uv (recommended)
    uv pip install -r requirements.txt

    # OR using standard pip
    # pip install -r requirements.txt
    ```

6.  **Configure Environment Variables:**
    * **Copy the example file:** `cp .env.example .env` (or copy manually).
    * **Edit `.env`:** Open the newly created `.env` file with a text editor.
    * **Fill in required values:**
        * `GOOGLE_API_KEY`: Add your Google API Key (required if using Gemini). Get one from [Google AI Studio](https://aistudio.google.com/app/apikey).
        * `ENTREZ_EMAIL`: Add your email address (required for PubMed Tool). NCBI uses this to identify requests.
    * **Configure LLMs:**
        * `DEFAULT_LLM_ID`: Set the default model the UI should use on startup (e.g., `gemini::gemini-1.5-flash`).
        * `GEMINI_AVAILABLE_MODELS`: List the Gemini models you want available in the UI dropdown, separated by commas (e.g., `gemini-1.5-flash,gemini-1.5-pro-latest`). Ensure these are accessible with your API key.
        * `OLLAMA_AVAILABLE_MODELS`: List the Ollama models you want available, separated by commas (e.g., `gemma:2b,llama3:latest`). Ensure these are pulled and running in your Ollama instance (`ollama list`).
        * `OLLAMA_BASE_URL`: Set the correct URL for your Ollama instance if you use it (e.g., `http://localhost:11434`).
    * **(Optional) Adjust Tuning & Settings:** Modify agent parameters (`AGENT_MAX_ITERATIONS`, `AGENT_MEMORY_WINDOW_K`, temperatures), tool settings (timeouts, limits, `TOOL_PDF_READER_WARNING_LENGTH`), server settings, or log level as needed. See comments in `.env.example` for details.
    * **Security:** The `.env` file is listed in `.gitignore` to prevent accidental commits of your secrets.

7.  **(If using Ollama)**
    * Install Ollama: [https://ollama.com/](https://ollama.com/)
    * Ensure the Ollama service is running.
    * **Important for Docker/WSL:** If Ollama runs as a systemd service, ensure it listens on all interfaces. Edit the service file (`sudo systemctl edit --full ollama.service`), add `Environment="OLLAMA_HOST=0.0.0.0"` under `[Service]`, then run `sudo systemctl daemon-reload` and `sudo systemctl restart ollama`.
    * Pull the models listed in `OLLAMA_AVAILABLE_MODELS`: `ollama pull <model_name>` (e.g., `ollama pull llama3:latest`).

## Running the Application

### Using Docker (Recommended Method)

Runs the backend server inside an isolated Docker container. **Highly recommended** for security and dependency management.

1.  **Prerequisites:** Ensure Docker and Docker Compose are installed.
2.  **Build and Run Backend:** From the project root directory (`ResearchAgent/`), run:
    ```bash
    docker compose up --build
    ```
    * The `--build` flag is needed the first time or after changing `Dockerfile` or `requirements.txt`.
    * Uses `network_mode: host` in `docker-compose.yml`, meaning the container shares the host network. The backend listens directly on host ports 8765 (WebSocket) and 8766 (File Server). Ensure these ports are free. This simplifies connecting to services like Ollama running directly on the host/WSL (use `http://localhost:11434` for `OLLAMA_BASE_URL`).
    * Keep this terminal running. Use `Ctrl+C` to stop.
3.  **Start Frontend Server:** Docker Compose only runs the backend. Serve the frontend files (HTML, CSS, JS) from a ***separate*** terminal in the project root:
    ```bash
    python3 -m http.server 8000
    ```
    * Keep this second terminal running.
4.  **Access the UI:** Open your web browser to `http://localhost:8000`.

**Development Workflow with Docker:**

* **Code Changes:** Changes to `./backend` code are reflected inside the container. Stop (`Ctrl+C`) and restart (`docker compose up`) the container to apply backend changes.
* **Dependency Changes:** If `requirements.txt` changes, rebuild with `docker compose up --build`.
* **Workspace & Database:** `./workspace` and `./database` are mounted as volumes, so data persists locally.

### Alternative: Running Directly on Host (Advanced / Less Secure)

**Not recommended** due to security risks of `Python_REPL` and `python_package_installer` executing directly in your host environment. **Proceed with extreme caution.**

1.  **Setup Environment:** Ensure Python 3.12+ is installed, activate a virtual environment (e.g., `uv venv`), and install dependencies (`uv pip install -r requirements.txt`).
    ```bash
    # Example activation (Linux/Mac/WSL)
    source .venv/bin/activate
    ```
2.  **Terminal 1: Start Backend Server:**
    ```bash
    python3 -m backend.server
    ```
3.  **Terminal 2: Start Frontend Server:**
    ```bash
    python3 -m http.server 8000
    ```
4.  **Access the UI:** Open `http://localhost:8000`.

## Usage & Testing

* **Create Task:** Click "+ New Task".
* **Rename Task:** Hover over a task, click the pencil icon (✏️).
* **Select Task:** Click a task to load its history.
* **Select LLM:** Use the dropdown in the chat header to choose the model for the current session.
* **Chat:** Interact with the agent. Use Up/Down arrows for input history. A "Thinking..." status line appears during processing.
* **Monitor:** Observe agent logs and status indicator (Idle/Running/Error/Disconnected).
* **Cancel Agent:** Click the "STOP" button in the monitor header while the agent is running to request cancellation (interrupts between steps).
* **Upload Files:** Select a task, click the "Upload File(s)" button below the task list, choose files. Check monitor/artifacts for confirmation.
* **Artifact Viewer:** View generated/uploaded images/text files using Prev/Next buttons.
* **Token Usage:** Observe token counts for the last LLM call and the current task total in the left panel.
* **Test PubMed Search:** Ask: `"Search PubMed for recent articles on CRISPR gene editing."`
* **Test Package Installation:** Ask: `"Install the 'numpy' python package."`
* **Test Python REPL:** Ask: `"Use the Python REPL tool to calculate 15 factorial."`
* **Test Image Generation:** Ask: `"Write a python script named 'plot.py' that uses matplotlib to create a simple sine wave plot and saves it as 'sine_wave.png'. Then execute the script using python."` (Ensure `matplotlib` is installed first).
* **Test File/PDF Reading:** Ask: `"Read the file named 'my_document.txt'"` or `"Read the file named 'research_paper.pdf'"` (assuming these files exist in the task workspace). Observe if a warning about length is added for large PDFs.
* **Test LLM Switching:** Select one model, ask a question. Select a different model, ask another question. Observe the agent's responses and potentially different styles.
* **Delete Task:** Click the trash icon (🗑️) next to a task (confirmation required).

## Known Issues

* **Agent Cancellation (STOP Button):** The STOP button sends a cancellation request to the agent. However, this typically interrupts the agent *between* major steps (like before the next tool use or LLM call) rather than immediately halting a long-running internal process within a tool or LLM.
* **Markdown Rendering in Chat:** The current Markdown rendering in the chat is basic. Complex Markdown or specific edge cases (e.g., underscores in filenames not intended for italics) might not always render as expected. Full GFM support via a dedicated library is a future enhancement.
* **Ollama Token Counts:** Token count reporting (specifically input tokens) for Ollama models can be less precise or consistently available compared to API-based models like Gemini. The displayed totals will reflect the information made available by the Ollama integration.

## Security Warnings

* **`python_package_installer` Tool:** Installs packages directly into the backend server's Python environment. **Significant security risk if exposed.**
* **`PythonREPLTool` Tool:** Executes arbitrary Python code directly in the backend server's environment. **Significant security risk if exposed.**
* **Recommendation:** **Strongly consider running the backend server inside a Docker container**, especially when using the `python_package_installer` or `PythonREPLTool`, to isolate execution and mitigate risks. Do not expose the backend ports directly to the internet without proper authentication and authorization layers.

## Future Perspectives & Ideas

* **Advanced Agent Architecture (Planner-Controller-Executor-Evaluator):**
    * **Planner:** Decomposes complex user requests into a sequence of smaller, manageable sub-tasks. This would likely involve an LLM call with a prompt designed for structured plan generation.
    * **Controller/Validator:** Before execution, this component would check the planned steps (e.g., ReAct-style "Action: tool_name, Action Input: ...") for validity. It would ensure the tool exists, the input format is likely correct, and there's no extraneous text or obvious errors. This could involve programmatic checks and/or a focused LLM call for syntax/tool validation.
    * **Executor:** Takes each validated sub-task and executes it. This could still leverage the current ReAct agent logic but focused on the specific sub-task goal.
    * **Evaluator:** After all sub-tasks are attempted, this component assesses the overall outcome against the initial user request. It could trigger re-planning if the task isn't complete or correct, providing feedback to the Planner.
    * This architecture aims to improve robustness for complex, multi-step tasks by introducing explicit planning, pre-execution validation, and post-execution evaluation, making it suitable for long-running research workflows.
* **Interactive Workflows & User Guidance:**
    * Allow the agent to request user confirmation or clarification at critical steps in a complex workflow.
    * Enable users to intervene, modify the agent's plan, or steer its execution if intermediate results are not as expected.
    * This makes the agent more of a collaborative partner, especially for long or exploratory tasks.
* **Automated Result Presentation & Website Generation:**
    * Empower the agent to generate simple, structured HTML pages or Markdown reports to present its findings, including text summaries, tables, and generated plots.
    * Potentially offer a few predefined design templates/CSS schemes for these reports to ensure consistent and professional-looking output.
* **Agent Control:** Improve the STOP button to more reliably and immediately halt ongoing agent tasks, potentially by running the agent in a separate process.
* **Streaming Output (Final Answer):** Implement token-by-token streaming for the agent's final answer to the UI for better responsiveness (currently, only the "Thinking..." line updates dynamically).
* **Drag & Drop Upload:** Enhance file upload to support drag and drop onto the task list or chat area.
* **Collapsible/Resizable Panels:** Allow the left task panel to be collapsed and/or the panel dividers to be dragged to resize areas.
* **PDF Reading Enhancements:** Add options for page ranges or user-specified `max_length` override, improve handling of complex layouts/images.
* **Structured Output:** Add capability for the agent to return results in structured formats (e.g., JSON) for easier parsing or downstream use (beyond just website generation).
* **Data Visualization:** Enable the agent (e.g., via Python REPL) to generate plots from data and display them directly in the Artifact Viewer.
* **Domain-Specific Tools:** Integrate more tools relevant to bioinformatics/epidemiology (e.g., BLAST, Ensembl/UniProt API access, VCF/FASTA parsing).
* **Artifact Management:** Allow users to rename or delete artifacts from the UI.
* **Chat Interaction Enhancements:**
    * Explore using special characters (e.g., `#toolname`) to suggest or force specific tool usage.
    * Optionally have the agent summarize its plan before execution for complex tasks (related to the Planner idea).
    * Consider restructuring the agent's chat bubble to clearly delineate Thought, Action, Observation, and Final Answer components for improved readability.
* **More Robust Formatting:** Use a dedicated Markdown library (e.g., `markdown-it` or `react-markdown` with plugins) for more complex rendering if needed.
* **Per-Task LLM Preference:** Store the last used LLM for each task.
* **Tool Configuration UI:** Allow enabling/disabling tools per task or globally.
* **User Authentication:** Add user accounts and login for multi-user scenarios.
* **UI Polish & Error Handling:** Continuously improve visual feedback and handle edge cases more gracefully.
* **Session Saving/Loading:** Persist the agent's memory state across browser sessions.
* **Cost/Token Tracking:** (Implemented) Display token usage for LLM calls.
