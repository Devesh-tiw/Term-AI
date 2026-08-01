#  Multi-Model AI Agent CLI

A sleek, asynchronous, and deeply integrated terminal-based AI assistant. Built entirely in Python, this CLI application leverages **Textual** for a highly responsive Terminal User Interface (TUI) and **LangChain** to orchestrate seamless conversations with cutting-edge Large Language Models via the **OpenRouter API**.

---

##  See it in Action

<video src="/home/devesh/ai-terminal-app/AI-Terminal.mp4" width="100%" controls autoplay loop muted></video>


---

##  System Architecture

This application is built with a modular, asynchronous architecture designed to prevent terminal freezing while maintaining a persistent conversational state. Here is exactly what happens under the hood, from the smallest component to the highest layer:

### 1. The Presentation Layer (Textual TUI)
* **Widgets & Layout:** The interface is constructed using Textual's CSS-like grid system. It utilizes `VerticalScroll` for the chat history, a `Markdown` widget to natively render code blocks and formatting from the AI, a `Select` dropdown for model routing, and an `Input` field for prompts.
* **Event Loop & Concurrency:** When a prompt is submitted (`on_input_submitted`), the application fires off an asynchronous task using Textual's `@work(exclusive=True)` decorator. This spins the LangChain API call into a separate worker thread, allowing the terminal UI to remain fully interactive and responsive (no freezing) while waiting for the AI to think.

### 2. The Orchestration Layer (LangChain)
* **Message Formatting:** User inputs are instantly converted into LangChain `HumanMessage` objects, while system instructions are maintained as `SystemMessage` objects.
* **In-Memory State Management:** The application is stateful per session. A Python list (`self.chat_history`) acts as the active RAM. Every user prompt and AI response (`AIMessage`) is appended to this list. The entire list is sent to the API on every turn, giving the LLM full conversational context. 
* **State Reset:** Triggering the `action_clear` function (via `Ctrl + L`) instantly flushes this Python list, resetting the AI's memory and clearing the screen without needing to restart the application.

### 3. The Routing & API Layer (OpenRouter)
* **Dynamic Slugs:** The application uses the `langchain-openrouter` dedicated package. When a user selects a model from the TUI dropdown, the specific model slug (e.g., `google/gemma-4-31b-it:free`) is injected directly into the `ChatOpenRouter` client. 
* **Cross-Platform Security:** API keys are never hardcoded. The application relies on `python-dotenv` to pull the `OPENROUTER_API_KEY` from a hidden `.env` file, ensuring complete security when pushing code to version control on both Windows and Linux.

---

##  Dependencies

This project relies on a lightweight stack of modern Python libraries.

| Package | Purpose |
| :--- | :--- |
| **`textual`** | The core framework powering the mouse-compatible, responsive terminal UI. |
| **`langchain`** | The orchestration framework used to manage LLM interactions and memory. |
| **`langchain-core`** | Provides the standard `SystemMessage`, `HumanMessage`, and `AIMessage` classes. |
| **`langchain-openrouter`** | The official integration package to connect LangChain seamlessly to OpenRouter. |
| **`python-dotenv`** | Cross-platform environment variable management for secure API key loading. |

---

##  Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/Devesh-tiw/ai-terminal-app.git](https://github.com/Devesh-tiw/ai-terminal-app.git)
cd ai-terminal-app
