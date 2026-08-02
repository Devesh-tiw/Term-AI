#  ai-terminal-app

> A sleek, asynchronous, terminal-native AI assistant — engineered as a proper installable CLI tool, not just a script. Built with **Textual**, orchestrated by **LangChain**, and routed through **OpenRouter**.

---

## 1.  Project Overview

**ai-terminal-app** is a professionally packaged, terminal-based AI assistant that brings large language model conversations directly into your shell. Under the hood, it combines a fully asynchronous **Textual** TUI with a **LangChain**-orchestrated conversation layer, dispatching every prompt through the **OpenRouter API** to access a rotating catalog of free and premium models.

This isn't a one-off script wrapped in a shell file — it's a `pip`-installable Python package, built around `pyproject.toml`, that registers itself as a real command-line executable. Once installed, the assistant launches from anywhere on your system with a single word: `ai-agent`.

✨ **Key highlights:**
- Fully async architecture — the TUI never blocks, even mid-generation
- Installable, packaged CLI tool (no more `./launch.sh` or `launch.bat`)
- Markdown-rendered responses directly in the terminal
- Dynamic model routing via OpenRouter's model slug system
- Secure `.env`-based credential management
- Mouse + keyboard driven Textual interface

---

## 2.  See it in Action

<div align="center">

<!--
  📌 Drag and drop your demo video or GIF (e.g. AI-Terminal-Fixed.mp4) directly
  onto this spot using the GitHub web editor. Do this from the "Edit" pane on
  github.com — GitHub will upload it to its asset CDN and auto-insert a
  playable embed link. Make sure the video is H.264-encoded so it renders
  correctly inline (older codecs like MPEG-4 ASP will show a blank frame).
-->

📌 **[Drop your demo video/GIF here — see comment above for instructions]**

</div>

<p align="center"><em>A quick walkthrough of ai-terminal-app: launching via <code>ai-agent</code>, streaming a Markdown-rendered response, and clearing memory with <code>Ctrl + L</code>.</em></p>

---

## 3.  System Architecture

ai-terminal-app follows a **modular, asynchronous, three-layer architecture**. Each layer owns a single responsibility, communicates through async boundaries, and can be extended or swapped independently.

```
┌───────────────────────────────────────────┐
│            Presentation Layer               │
│            Textual TUI                      │
│  • Markdown rendering of AI responses        │
│  • Async @work tasks — zero UI blocking      │
│  • Mouse + keyboard driven navigation        │
└───────────────────┬───────────────────────────┘
                    │ async event dispatch
┌───────────────────▼───────────────────────────┐
│            Orchestration Layer                 │
│            LangChain                            │
│  • In-memory conversation state management       │
│  • Prompt templating & chat history assembly       │
└───────────────────┬───────────────────────────┘
                    │ HTTPS requests
┌───────────────────▼───────────────────────────┐
│            Routing Layer                        │
│            OpenRouter API                        │
│  • Dynamic model slug resolution                   │
│  • Routes to free/premium LLM backends              │
└───────────────────────────────────────────────┘
```

### 🖥️ Presentation Layer — Textual
The UI is built entirely on Textual's reactive, async-first framework. AI responses are rendered as live Markdown (headings, code blocks, lists) rather than raw text. Every network-bound operation — most importantly, the call out to OpenRouter — runs inside a Textual `@work` worker task. This keeps long-running generations off the main event loop, so the interface stays fully interactive (scrolling, typing, mouse clicks) while a response streams in.

### 🧩 Orchestration Layer — LangChain
LangChain manages the conversational brain of the app: assembling prompt templates, maintaining in-memory chat history for the current session, and structuring the request payload sent downstream. This abstraction means swapping prompt strategies or memory backends later (see Roadmap) won't require touching the UI layer at all.

### 🌐 Routing Layer — OpenRouter
Rather than hardcoding a single model, requests are routed through OpenRouter using dynamic model slugs (e.g. switching between different free-tier models at runtime). This keeps the assistant flexible, cost-free by default, and resilient if any individual upstream model is rate-limited or deprecated.

---

## 4. 📦 Dependencies

| Package               | Purpose                                                         |
|------------------------|--------------------------------------------------------------------|
| `textual`              | Powers the asynchronous, mouse-and-keyboard-driven terminal UI      |
| `langchain`             | Orchestrates prompts, chains, and conversational memory              |
| `langchain-core`        | Core abstractions and primitives used throughout LangChain             |
| `langchain-openrouter`  | Integration layer connecting LangChain to the OpenRouter API           |
| `python-dotenv`         | Loads API credentials securely from a local `.env` file                 |

---

## 5. ⚙️ Modern Installation & Setup

This project has moved past shell/batch launch scripts. It's now a proper Python package defined by `pyproject.toml`, meaning it installs like any other CLI tool and registers a real executable on your system.

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/ai-terminal-app.git
cd ai-terminal-app
```

### Step 2 — Create and activate a virtual environment

**🐧 Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

**🪟 Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### Step 3 — Configure your environment variables 🔐

Copy the provided example file and fill in your own OpenRouter API key:

```bash
cp .env.example .env       # Linux/macOS
copy .env.example .env     # Windows
```

Open `.env` and add your key:

```env
OPENROUTER_API_KEY=your_api_key_here
```

> ⚠️ `.env` is git-ignored by default — never commit real credentials. `python-dotenv` loads this file automatically at runtime via `load_dotenv()`.

### Step 4 — Install the package (the new way) 📦

Instead of running a script, the app is now installed as an **editable package** using `pyproject.toml`:

```bash
pip install -e .
```

The `-e` (editable) flag installs the project in development mode — pointing directly at your local source — while `pip` reads `pyproject.toml` to register the console entry point.

### Step 5 — Launch it from anywhere 

Once installed, forget `python ai_app.py` or `./launch.sh` entirely. The package registers a global command:

```bash
ai-agent
```

You can now run `ai-agent` from **any directory**, in any shell session, as long as your virtual environment is active.

### 🌍 (Optional) True global installation with pipx

If you'd like `ai-agent` available system-wide, outside of any virtual environment, use [`pipx`](https://pypa.github.io/pipx/):

```bash
pipx install .
```

`pipx` installs the package into its own isolated environment while still exposing the `ai-agent` command globally — giving you a real, standalone OS-level CLI tool.

---

## 6. ⌨️ Keybindings & Usage

Launch the assistant with `ai-agent` and start chatting immediately.

| Input               | Action                                        |
|----------------------|--------------------------------------------------|
| `Enter`              | Submit your prompt to the AI                       |
| `Ctrl + L`            | Clear conversation memory                           |
| `Ctrl + C`             | Quit the application                                  |
| 🖱️ Mouse              | Scroll response history, click to focus input, and navigate the UI |

Responses stream in and render live as Markdown — code blocks, headings, and lists all display formatted, right in your terminal.

---

## 7. 🛣️ Roadmap

ai-terminal-app is under active development. Planned upgrades include:

- 🎙️ **Voice Engine (STT/TTS)** — Local **Whisper**-based speech-to-text for hands-free prompting, paired with text-to-speech output, so the assistant becomes fully voice-interactive without relying on cloud speech APIs.
- 💾 **Permanent Database Memory** — Replacing in-memory-only conversation state with a persistent **SQLite + LangChain memory** backend, so context and chat history survive across sessions and restarts.

Contributions, issues, and feature requests are always welcome. 🌟

---

<p align="center">
  <strong>Developed by Devesh Tiwari</strong>
</p>