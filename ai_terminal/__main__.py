
import os
import asyncio
import base64
import subprocess
from pathlib import Path
from open_router import get_available_models
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.dirname(current_dir)
load_dotenv(os.path.join(root_dir, ".env"))

from textual.app        import App, ComposeResult
from textual.containers import VerticalScroll, Container, Horizontal, Vertical
from textual.widgets    import (
    Header, Footer, Input, Markdown, Static, Select,
    Button, TextArea, Collapsible, Switch, Label,
)
from textual            import work
from textual.reactive   import reactive

from langchain_openrouter           import ChatOpenRouter
from langchain_core.messages        import SystemMessage, HumanMessage, AIMessage

try:
    import httpx
    from bs4 import BeautifulSoup
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False



class LocalAgent:
    """
    Three-tool agent that Claude can call autonomously or the user can
    invoke directly with slash commands.

    Commands:
        /shell  <cmd>           – run a shell command and return stdout/stderr
        /read   <path>          – read a file from disk
        /write  <path> <text>   – write text to a file
        /fetch  <url>           – fetch a URL and return cleaned text
    """

    @staticmethod
    async def run_shell(command: str) -> str:
        """Run a shell command asynchronously, return combined output."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            out = stdout.decode(errors="replace").strip()
            err = stderr.decode(errors="replace").strip()
            parts = []
            if out:
                parts.append(f"**stdout:**\n```\n{out}\n```")
            if err:
                parts.append(f"**stderr:**\n```\n{err}\n```")
            if not parts:
                parts.append(f"*(Process exited with code {proc.returncode} — no output)*")
            return "\n\n".join(parts)
        except asyncio.TimeoutError:
            return "❌ **Shell timeout** — command took longer than 30 s."
        except Exception as e:
            return f"❌ **Shell error:** {e}"

    @staticmethod
    async def read_file(path: str) -> str:
        """Read a file from disk."""
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return f"❌ File not found: `{p}`"
            if p.stat().st_size > 5 * 1024 * 1024:          # 5 MB guard
                return f"❌ File too large to read (> 5 MB): `{p}`"
            content = p.read_text(errors="replace")
            lang = p.suffix.lstrip(".") or "text"
            return f"**File:** `{p}`\n\n```{lang}\n{content}\n```"
        except Exception as e:
            return f"❌ **Read error:** {e}"

    @staticmethod
    async def write_file(path: str, text: str) -> str:
        """Write text to a file, creating parent dirs as needed."""
        try:
            p = Path(path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            return f"✅ Written **{len(text):,}** chars → `{p}`"
        except Exception as e:
            return f"❌ **Write error:** {e}"

    @staticmethod
    async def fetch_url(url: str) -> str:
        """Fetch a URL and return cleaned plain text (no JS required)."""
        if not WEB_AVAILABLE:
            return "❌ `httpx` and `beautifulsoup4` are required for web fetch. Run: `pip install httpx beautifulsoup4`"
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
                resp = await client.get(url, headers={"User-Agent": "ai-terminal-agent/2.0"})
                resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Trim to ~8000 chars to avoid blowing context
            if len(text) > 8000:
                text = text[:8000] + "\n\n*[truncated — content exceeds 8 000 chars]*"
            return f"**URL:** {url}\n\n```\n{text}\n```"
        except Exception as e:
            return f"❌ **Fetch error:** {e}"



class AITerminalApp(App):

    # ── Key bindings ──────────────────────────────────────────────────────────
    BINDINGS = [
        ("ctrl+c",  "quit",         "Quit"),
        ("ctrl+l",  "clear",        "Clear Memory"),
        ("ctrl+p",  "toggle_panel", "Input Panel"),
        ("ctrl+a",  "toggle_agent", "Agent Mode"),
    ]

    # ── CSS ───────────────────────────────────────────────────────────────────
    CSS = """
    /* ── Layout ─────────────────────────────────────────── */
    Screen {
        background: $surface;
        layout: vertical;
    }

    /* ── Chat output ─────────────────────────────────────── */
    #output-container {
        height: 1fr;
        border: solid $accent;
        padding: 1 2;
        margin: 1 1 0 1;
    }

    /* ── Rich input panel ────────────────────────────────── */
    #input-panel {
        height: auto;
        max-height: 14;
        margin: 0 1;
        border: dashed $primary;
        padding: 0 1;
    }
    #panel-header {
        color: $accent;
        text-style: bold;
        padding: 0 1;
    }
    #panel-type-row {
        height: 3;
        margin-bottom: 0;
    }
    #panel-type-label {
        width: auto;
        padding: 1 1 0 0;
        color: $text-muted;
    }
    #content-type-selector {
        width: 40%;
    }
    #paste-area {
        height: 8;
        border: solid $surface-darken-2;
        margin: 0;
    }
    #panel-hint {
        color: $text-muted;
        padding: 0 1 1 1;
    }

    /* ── Bottom control bar ──────────────────────────────── */
    #input-area {
        height: auto;
        margin: 0 1 1 1;
    }
    #controls-row {
        height: 3;
        margin-bottom: 1;
    }
    #model-selector {
        width: 52%;
    }
    #agent-toggle-label {
        width: auto;
        padding: 1 1 0 1;
        color: $text-muted;
    }
    #agent-toggle {
        margin-top: 1;
    }
    #status-label {
        color: $text-muted;
        padding: 1 0 0 2;
        width: 1fr;
    }
    #prompt-input {
        border: tall $primary;
    }

    /* ── Agent mode indicator ────────────────────────────── */
    .agent-active {
        color: $success;
        text-style: bold;
    }
    """


    AVAILABLE_MODELS = get_available_models()

    CONTENT_TYPES = [
        ("💬 Plain text / question",  "text"),
        ("🖼️  Image file path",        "image"),
        ("📄  Document / long text",   "document"),
        ("🗂️  File path → read & send", "filepath"),
        ("💻  Code snippet",           "code"),
    ]

    agent_mode: reactive[bool] = reactive(False)
    panel_open: reactive[bool] = reactive(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat_history: list = [
            SystemMessage(content=(
                "You are a powerful CLI terminal assistant. "
                "You have access to local agent tools: shell commands, file read/write, and web fetch. "
                "When agent mode is ON, you may suggest using these tools. "
                "Keep responses concise and precise. Format code in fenced blocks."
            ))
        ]
        self.agent = LocalAgent()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Chat output
        yield VerticalScroll(
            Markdown(
                "### 🤖 AI Terminal  v2.0\n"
                "**New:** 5 powerful free models · Rich input panel · Local agent tools\n\n"
                "| Shortcut | Action |\n"
                "|----------|--------|\n"
                "| `Ctrl+P` | Open / close input panel |\n"
                "| `Ctrl+A` | Toggle agent mode |\n"
                "| `Ctrl+L` | Clear memory |\n"
                "| `/shell <cmd>` | Run a shell command |\n"
                "| `/read <path>` | Read a file |\n"
                "| `/write <path> <text>` | Write to a file |\n"
                "| `/fetch <url>` | Fetch a web page |",
                id="ai-response",
            ),
            id="output-container",
        )

        with Vertical(id="input-panel"):
            yield Static("📎  Rich Input Panel  (Ctrl+P to toggle)", id="panel-header")
            yield Horizontal(
                Static("Type:", id="panel-type-label"),
                Select(
                    self.CONTENT_TYPES,
                    prompt="Content type",
                    value="text",
                    id="content-type-selector",
                ),
                id="panel-type-row",
            )
            yield TextArea(
                "",
                id="paste-area",
                language="markdown",
                show_line_numbers=False,
            )
            yield Static(
                "Paste code, text, an image path, or a file path here. "
                "It will be injected with your prompt when you press Enter.",
                id="panel-hint",
            )

        yield Container(
            Horizontal(
                Select(
                    self.AVAILABLE_MODELS,
                    prompt="Select a Model",
                    value=self.AVAILABLE_MODELS[0][1],
                    id="model-selector",
                ),
                Static("Agent:", id="agent-toggle-label"),
                Switch(value=False, id="agent-toggle"),
                Static("Ready.", id="status-label"),
                id="controls-row",
            ),
            Input(
                placeholder="Ask something… or /shell ls -la  /read ./file.py  /fetch https://…",
                id="prompt-input",
            ),
            id="input-area",
        )

        yield Footer()

    def on_mount(self) -> None:
        # Hide the panel initially
        self.query_one("#input-panel").display = False

    def action_toggle_panel(self) -> None:
        panel = self.query_one("#input-panel")
        self.panel_open = not self.panel_open
        panel.display = self.panel_open

    def action_toggle_agent(self) -> None:
        toggle = self.query_one("#agent-toggle", Switch)
        toggle.value = not toggle.value

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self.agent_mode = event.value
        status = self.query_one("#status-label", Static)
        if self.agent_mode:
            status.update("🤖 [bold green]Agent mode ON[/] — AI can use local tools")
        else:
            status.update("Ready.")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_prompt = event.value.strip()
        if not user_prompt:
            return

        # Grab model
        model_selector  = self.query_one("#model-selector", Select)
        selected_model  = model_selector.value
        if not selected_model:
            self.query_one("#status-label", Static).update("⚠️  Select a model first!")
            return

        # Clear prompt input
        self.query_one("#prompt-input", Input).value = ""

        if user_prompt.startswith("/"):
            self.run_agent_command(user_prompt)
            return

        full_prompt = await self._build_prompt(user_prompt)

        # Update status
        display_name = selected_model.split("/")[-1]
        agent_tag    = " [Agent]" if self.agent_mode else ""
        self.query_one("#status-label", Static).update(
            f"⏳ {display_name}{agent_tag} thinking…"
        )

        self.run_llm_query(full_prompt, user_prompt, selected_model)

    async def _build_prompt(self, user_prompt: str) -> str:
        if not self.panel_open:
            return user_prompt

        panel_text   = self.query_one("#paste-area", TextArea).text.strip()
        content_type = self.query_one("#content-type-selector", Select).value or "text"

        if not panel_text:
            return user_prompt

        if content_type == "filepath":
            # Read file from disk and inject content
            file_result = await self.agent.read_file(panel_text)
            return f"{user_prompt}\n\n---\n{file_result}"

        if content_type == "image":
            p = Path(panel_text).expanduser()
            if p.exists():
                try:
                    b64 = base64.b64encode(p.read_bytes()).decode()
                    # For vision models we embed a data URI note in the text
                    return (
                        f"{user_prompt}\n\n"
                        f"[Image attached: `{p.name}` — base64 length {len(b64)} chars. "
                        f"Describe or analyse as requested.]"
                    )
                except Exception:
                    pass
            return f"{user_prompt}\n\n[Image path given: `{panel_text}` — file not readable]"

        if content_type == "code":
            lang = Path(panel_text.splitlines()[0]).suffix.lstrip(".") if panel_text else ""
            return f"{user_prompt}\n\n```{lang}\n{panel_text}\n```"

        if content_type == "document":
            return f"{user_prompt}\n\n---\n**Document:**\n{panel_text}"

        # default: plain text
        return f"{user_prompt}\n\n---\n{panel_text}"

    @work(exclusive=False)
    async def run_agent_command(self, command: str) -> None:
        markdown = self.query_one("#ai-response", Markdown)
        status   = self.query_one("#status-label", Static)

        parts = command.split(maxsplit=2)
        verb  = parts[0].lower()

        status.update(f"⚙️  Running `{verb}`…")

        if verb == "/shell":
            cmd    = " ".join(parts[1:]) if len(parts) > 1 else ""
            result = await self.agent.run_shell(cmd) if cmd else "❌ Usage: `/shell <command>`"

        elif verb == "/read":
            path   = parts[1] if len(parts) > 1 else ""
            result = await self.agent.read_file(path) if path else "❌ Usage: `/read <path>`"

        elif verb == "/write":
            if len(parts) < 3:
                result = "❌ Usage: `/write <path> <text>`"
            else:
                result = await self.agent.write_file(parts[1], parts[2])

        elif verb == "/fetch":
            url    = parts[1] if len(parts) > 1 else ""
            result = await self.agent.fetch_url(url) if url else "❌ Usage: `/fetch <url>`"

        else:
            result = f"❌ Unknown command `{verb}`. Available: `/shell` `/read` `/write` `/fetch`"

        markdown.update(f"### ⚙️ Agent Command: `{command}`\n\n---\n\n{result}")
        status.update("✅ Done.")

    @work(exclusive=True)
    async def run_llm_query(self, full_prompt: str, display_prompt: str, model_id: str) -> None:
        markdown = self.query_one("#ai-response", Markdown)
        status   = self.query_one("#status-label", Static)

        markdown.update(
            f"### 💬 You:\n{display_prompt}\n\n---\n\n*Thinking…*"
        )

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            markdown.update("❌ **Error:** `OPENROUTER_API_KEY` not set in `.env`.")
            status.update("❌ Missing API Key.")
            return

        try:
            llm = ChatOpenRouter(
                model=model_id,
                api_key=api_key,
                temperature=0.2,
            )

            user_msg_content = full_prompt
            if self.agent_mode:
                user_msg_content = (
                    "[AGENT MODE] You may recommend the user run these commands if helpful:\n"
                    "  /shell <cmd>     — execute a shell command\n"
                    "  /read  <path>    — read a file\n"
                    "  /write <path> <text> — write a file\n"
                    "  /fetch <url>     — fetch a web page\n\n"
                ) + user_msg_content

            self.chat_history.append(HumanMessage(content=user_msg_content))
            response = await llm.ainvoke(self.chat_history)
            self.chat_history.append(response)

            markdown.update(
                f"### 💬 You:\n{display_prompt}\n\n---\n\n{response.content}"
            )
            status.update("✅ Done.")

        except Exception as e:
            markdown.update(
                f"### 💬 You:\n{display_prompt}\n\n---\n\n"
                f"❌ **Error:**\n```\n{str(e)}\n```"
            )
            status.update("❌ Error.")
    def action_clear(self) -> None:
        self.chat_history = [
            SystemMessage(content=(
                "You are a powerful CLI terminal assistant with local agent tools. "
                "Keep responses concise and precise."
            ))
        ]
        self.query_one("#ai-response", Markdown).update(
            "### 🧹 Memory cleared.\nReady for your next prompt."
        )
        self.query_one("#status-label", Static).update("Ready.")
        # Also clear the paste panel
        try:
            self.query_one("#paste-area", TextArea).load_text("")
        except Exception:
            pass


def main():
    app = AITerminalApp()
    app.run()

if __name__ == "__main__":
    main()