#  AI-terminal-app

> A sleek, asynchronous, terminal-native AI assistant — engineered as a proper installable CLI tool, not just a script. Built with **Textual**, orchestrated by **LangChain**, routed through **OpenRouter**, and now voice-enabled end to end.

---

## 1.  Project Overview

**ai-terminal-app** is a professionally packaged, terminal-based AI assistant that brings large language model conversations directly into your shell. Under the hood, it combines a fully asynchronous **Textual** TUI with a **LangChain**-orchestrated conversation layer, dispatching every prompt through the **OpenRouter API** to access a rotating catalog of free and premium models.

This isn't a one-off script wrapped in a shell file — it's a `pip`-installable Python package, built around `pyproject.toml`, that registers itself as a real command-line executable. Once installed, the assistant launches from anywhere on your system with a single word: `ai-agent`.

✨ **Key highlights:**
- Fully async architecture — the TUI never blocks, even mid-generation
- Installable, packaged CLI tool (no more `./launch.sh` or `launch.bat`)
- Markdown-rendered responses directly in the terminal, with full conversation history preserved on screen
- Dynamic model routing via OpenRouter's model slug system, filtered to text-chat-capable models only
- Voice in, voice out — local Whisper transcription (F5) and spoken responses via edge-tts (F6)
- Built-in agent tools (`/shell`, `/read`, `/write`, `/fetch`) for an optional Agent mode
- Secure `.env`-based credential management
- Mouse + keyboard driven Textual interface

---

## 2.  See it in Action

<div align="center">



https://github.com/user-attachments/assets/963743da-e99f-45b9-a973-7f7fde73c385



</div>

<p align="center"><em>A quick walkthrough of ai-terminal-app: launching via <code>ai-agent</code>, streaming a Markdown-rendered response, and clearing memory with <code>Ctrl + L</code>.</em></p>

<div align="center">

<!-- TODO: replace with your new v2.1 screenshot -->
<img src="./assets/ai-terminal-app-v2.1.png" alt="ai-terminal-app v2.1 screenshot" width="800"/>

</div>

<p align="center"><em>Updated look at v2.1 — voice controls, Agent mode, and the refreshed input panel.</em></p>

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
│  • Voice I/O: STT input (F5), TTS output (F6)│
└───────────────────┬───────────────────────────┘
                    │ async event dispatch
┌───────────────────▼───────────────────────────┐
│            Orchestration Layer                 │
│            LangChain                            │
│  • In-memory conversation state management       │
│  • Prompt templating & chat history assembly       │
│  • Agent tool dispatch (/shell, /read, /write, /fetch) │
└───────────────────┬───────────────────────────┘
                    │ HTTPS requests
┌───────────────────▼───────────────────────────┐
│            Routing Layer                        │
│            OpenRouter API                        │
│  • Dynamic model slug resolution                   │
│  • Text-chat-only filtering (excludes TTS/image/   │
│    music generation models from the picker)          │
│  • Routes to free/premium LLM backends              │
└───────────────────────────────────────────────┘
```

### 🖥️ Presentation Layer — Textual
The UI is built entirely on Textual's reactive, async-first framework. AI responses are rendered as live Markdown (headings, code blocks, lists) rather than raw text, and the full conversation transcript is re-rendered on every turn so nothing gets overwritten. Every network-bound operation — most importantly, the call out to OpenRouter — runs inside a Textual `@work` worker task, keeping the interface fully interactive (scrolling, typing, mouse clicks) while a response streams in.

### 🧩 Orchestration Layer — LangChain
LangChain manages the conversational brain of the app: assembling prompt templates, maintaining in-memory chat history for the current session, and structuring the request payload sent downstream. In Agent mode, it also dispatches slash-command tool calls. This abstraction means swapping prompt strategies or memory backends later (see Roadmap) won't require touching the UI layer at all.

### 🌐 Routing Layer — OpenRouter
Rather than hardcoding a single model, requests are routed through OpenRouter using dynamic model slugs (e.g. switching between different free-tier models at runtime). The model list is filtered through `_is_text_chat_model()`, which checks `architecture.output_modalities` for `"text"` and applies a regex safety net to exclude non-chat families (`lyria`, `veo`, `imagen`, `music`, `-tts`, `whisper`, `dall-e`, `stable-diffusion`) — so the dropdown only ever shows models that can actually hold a conversation.

---

## 4. 📦 Dependencies

| Package               | Purpose                                                         |
|------------------------|--------------------------------------------------------------------|
| `textual`              | Powers the asynchronous, mouse-and-keyboard-driven terminal UI      |
| `langchain`             | Orchestrates prompts, chains, and conversational memory              |
| `langchain-core`        | Core abstractions and primitives used throughout LangChain             |
| `langchain-openrouter`  | Integration layer connecting LangChain to the OpenRouter API           |
| `python-dotenv`         | Loads API credentials securely from a local `.env` file                 |

### Optional voice extras (`pip install -e ".[voice]"`)

| Package          | Purpose                                          |
|-------------------|---------------------------------------------------|
| `faster-whisper`  | Local speech-to-text transcription (F5)             |
| `sounddevice`      | Microphone audio capture                             |
| `numpy`             | Audio buffer processing for STT                        |
| `edge-tts`           | Text-to-speech playback of AI responses (F6)              |

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

### Step 4 — Install the package 📦

```bash
pip install -e .
```

The `-e` (editable) flag installs the project in development mode — pointing directly at your local source — while `pip` reads `pyproject.toml` to register the console entry point.

**Want voice features too?** Install with the optional `voice` extra, which pulls in `faster-whisper`, `sounddevice`, `numpy`, and `edge-tts`:

```bash
pip install -e ".[voice]"
```

> ℹ️ On first launch, the Whisper model preloads in a background thread (`on_mount`) so the F5 voice key is ready immediately instead of stalling on first use.

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

| Key            | Action                                                              |
|-----------------|--------------------------------------------------------------------------|
| `Enter`          | Submit your prompt to the AI                                              |
| `Ctrl + L`        | Clear conversation memory                                                  |
| `Ctrl + P`         | Toggle the rich input panel (paste code, docs, images, or file paths)        |
| `Ctrl + C`           | Quit the application                                                          |
| `F5`                  | Toggle voice recording — transcribes via local Whisper and sends automatically, no `Enter` needed |
| `F6`                    | Toggle text-to-speech — AI responses are spoken aloud via `edge-tts`             |
| `F7`                      | Toggle Agent mode (enables the slash-command tools below)                          |
| 🖱️ Mouse                   | Scroll response history, click to focus input, and navigate the UI                    |

> Note: Agent mode toggling moved from `Ctrl + A` to `F7` due to a terminal-emulator key conflict.

Responses stream in and render live as Markdown — code blocks, headings, and lists all display formatted, right in your terminal, with the full transcript preserved across turns.

### 🛠️ Agent Mode — Slash Commands

With Agent mode on (`F7`), the following tools are available directly from the input line:

| Command                 | Action                                   |
|---------------------------|---------------------------------------------|
| `/shell <cmd>`             | Run a shell command                            |
| `/read <path>`               | Read a file                                       |
| `/write <path> <text>`         | Write a file                                         |
| `/fetch <url>`                    | Fetch a webpage as cleaned text                         |

---

## 7. 🧾 What's New in v2.1

- **Fixed model loading** — `openrouter_models.py` now uses a relative-import-first, absolute-import-fallback pattern, resolving a silent failure that occurred under packaged execution.
- **Fixed chat memory not rendering** — the transcript now accumulates in `self.visible_transcript` and re-renders in full each turn, instead of each reply overwriting the previous one.
- **Fixed voice "stuck transcribing"** — the Whisper model now preloads at startup instead of lazy-loading on first `F5` press, plus a 25s hard timeout with a clear error if transcription genuinely stalls.
- **Fixed non-chat models appearing in the picker** — added `_is_text_chat_model()` plus a regex safety net to exclude music/image/TTS/whisper models (e.g. Lyria) that were slipping past the free-tier filter.
- **Voice now sends automatically** — transcription is wired into a shared `_submit_prompt()` method, so speaking a prompt no longer requires an extra `Enter` press.

---

## 8. 🛣️ Roadmap

ai-terminal-app is under active development. Planned upgrades include:

- 💾 **Permanent Database Memory** — Replacing in-memory-only conversation state with a persistent **SQLite + LangChain memory** backend, so context and chat history survive across sessions and restarts.
- 🖼️ **Real Multimodal Image Support** — Wiring image input properly into vision-capable models via proper multipart message construction, replacing the current placeholder that only embeds a base64-length note rather than the actual image content.

Contributions, issues, and feature requests are always welcome. 🌟

---

<p align="center">
  <strong>Developed by Devesh Tiwari</strong><br/>
  <em>Amrita Vishwa Vidyapeetham</em>
</p>