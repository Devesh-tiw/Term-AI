#  AI-Terminal — Your Command-Line AI Companion

> A sleek, asynchronous, terminal-native AI assistant that brings the power of large language models straight into your shell — no browser tabs, no context-switching, just you and your terminal.

---

## 1.  Project Overview

**AI-Terminal** is a Python-powered, terminal-based AI assistant built with **Textual** for a rich, responsive TUI and **LangChain** for intelligent conversation orchestration. It routes every prompt through the **OpenRouter API**, giving you free access to a rotating lineup of powerful language models — all without leaving your command line.

Whether you're debugging code, brainstorming ideas, or just want a fast AI chat interface that feels at home in a terminal, AI-Terminal delivers a fluid, non-blocking experience powered by modern async Python.

 **Key highlights:**
- Fully asynchronous — the UI never freezes, even mid-response
- Clean, keyboard-driven TUI built with Textual
- Model routing via OpenRouter's free-tier catalog
- Secure, `.env`-based API key management
- Minimal footprint, maximum terminal vibes

---

## 2.  See it in Action

<div align="center">


https://github.com/user-attachments/assets/eccc7859-d406-4802-87aa-9bb977e6d369

</div>








<br>

<p align="center"><em>A 31-second walkthrough of the AI-Terminal TUI in action — routing a prompt through OpenRouter, streaming a response, and clearing memory with <code>Ctrl + L</code>.</em></p>

---

## 3.  System Architecture

AI-Terminal is built on a **modular, asynchronous, three-layer architecture** designed to keep the interface responsive no matter how long a model takes to respond.

```
┌─────────────────────────────────────────┐
│         Presentation Layer               │
│         (Textual — async TUI)            │
│  Handles rendering, input, keybindings   │
└───────────────────┬───────────────────────┘
                    │ async calls
┌───────────────────▼───────────────────────┐
│         Orchestration Layer                │
│         (LangChain)                        │
│  Manages prompts, memory, chat history     │
└───────────────────┬───────────────────────┘
                    │ API requests
┌───────────────────▼───────────────────────┐
│         Routing Layer                      │
│         (OpenRouter API)                   │
│  Routes prompts to free LLM models         │
└─────────────────────────────────────────┘
```

### Presentation Layer (Textual)
Handles all rendering, layout, and keyboard input. Built entirely on Textual's async event loop, this layer captures user input and displays streaming AI responses without ever blocking the main thread.

###  Orchestration Layer (LangChain)
Acts as the brain of the operation — managing conversation memory, prompt templates, and chat history. LangChain abstracts away the complexity of talking to different LLM backends and keeps conversational context coherent across turns.

### Routing Layer (OpenRouter)
Routes each request to one of several free models available through OpenRouter, giving you flexibility and redundancy without vendor lock-in or API cost.

**Why async matters:** Because every network call (model inference) runs inside Python's `asyncio` event loop, **the terminal UI never freezes** — you can keep interacting with the interface, scroll through history, or even cancel a request while a response streams in.

---

## 4.  Dependencies

| Package               | Purpose                                                        |
|------------------------|------------------------------------------------------------------|
| `textual`              | Powers the asynchronous terminal user interface (TUI)           |
| `langchain`             | Orchestrates prompts, chains, and conversational memory          |
| `langchain-core`        | Core abstractions and primitives used by LangChain                |
| `langchain-openrouter`  | Integration layer connecting LangChain to the OpenRouter API      |
| `python-dotenv`         | Loads environment variables securely from a `.env` file           |

---

## 5.  Installation & Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/ai-terminal.git
cd ai-terminal
```

### Step 2 — Create a virtual environment

** Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

** Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

** Windows (Command Prompt)**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure your `.env` file 

AI-Terminal uses `python-dotenv` to keep your API credentials out of source code. Create a `.env` file in the project root:

```bash
touch .env       # Linux/macOS
type nul > .env  # Windows
```

Add your OpenRouter API key inside it:

```env
OPENROUTER_API_KEY=your_api_key_here
```

> **Never commit your `.env` file to version control.** Make sure it's listed in your `.gitignore`. The app loads this key automatically at runtime via `load_dotenv()`, keeping your credentials secure and out of the codebase.

### Step 5 — Launch the app

**Linux / macOS**
```bash
./launch.sh
```

**Windows**
```cmd
launch.bat
```

---

## 6.  Usage

Once launched, AI-Terminal drops you into an interactive chat interface. Use the following keybindings to navigate:

| Keybinding      | Action                              |
|------------------|--------------------------------------|
| `Enter`          | Submit your prompt to the AI          |
| `Ctrl + L`        | Clear conversation memory              |
| `Ctrl + C`        | Quit the application                    |

Simply type your message, hit `Enter`, and watch the response stream in — all without your terminal ever locking up.

---

## 7.  Roadmap

AI-Terminal is actively evolving. Here's what's coming next:

-  **Voice Engine (STT/TTS)** — Speak your prompts and hear responses read back, turning the terminal into a full voice-interactive assistant.
-  **Permanent Database Memory** — Persistent conversation history powered by SQLite and LangChain memory modules, so context survives across sessions.

Stay tuned — contributions and feature suggestions are always welcome! 🌟

---

<p align="center">Built with ❤️ for the terminal.</p>
