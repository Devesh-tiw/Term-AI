import os
import re
import asyncio
import base64
import tempfile
import threading
import webbrowser
import platform
import subprocess
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.dirname(current_dir)
load_dotenv(os.path.join(root_dir, ".env"))

from textual.app        import App, ComposeResult
from textual.containers import VerticalScroll, Container, Horizontal, Vertical
from textual.screen     import Screen, ModalScreen
from textual.widgets    import (
    Header, Footer, Input, Markdown, Static, Select,
    TextArea, Switch, Button,
)
from textual            import work
from textual.reactive   import reactive

OPENROUTER_KEYS_URL = "https://openrouter.ai/keys"

from langchain_openrouter    import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
try:
    import httpx
    from bs4 import BeautifulSoup
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import edge_tts
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from .open_router import get_available_models, aget_available_models
    DYNAMIC_MODELS = True
except ImportError:
    try:
        from .open_router import get_available_models, aget_available_models
        DYNAMIC_MODELS = True
    except ImportError:
        DYNAMIC_MODELS = False


class VoiceEngine:
    WHISPER_MODEL_SIZE = "tiny"
    SAMPLE_RATE        = 16_000
    TTS_VOICE          = "en-US-AriaNeural"

    def __init__(self):
        self._whisper: "WhisperModel | None" = None
        self._recording       = False
        self._recorded_frames: list = []
        self._tts_enabled     = False
        self._tts_proc = None 
        self._record_lock     = threading.Lock()
        self.model_ready      = False

    def preload_whisper(self) -> None:
        if not WHISPER_AVAILABLE:
            return
        if self._whisper is None:
            self._whisper = WhisperModel(
                self.WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
        self.model_ready = True

    def _get_whisper(self):
        if self._whisper is None:
            if not WHISPER_AVAILABLE:
                raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")
            self._whisper = WhisperModel(
                self.WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
            self.model_ready = True
        return self._whisper

    def start_recording(self) -> bool:
        if not AUDIO_AVAILABLE:
            return False
        with self._record_lock:
            if self._recording:
                return False
            self._recording = True
            self._recorded_frames = []

        def _callback(indata, frames, time_info, status):
            if self._recording:
                self._recorded_frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()
        return True

    def stop_recording(self) -> "np.ndarray | None":
        if not AUDIO_AVAILABLE:
            return None
        with self._record_lock:
            self._recording = False
        self._stream.stop()
        self._stream.close()
        if not self._recorded_frames:
            return None
        return np.concatenate(self._recorded_frames, axis=0).flatten()

    async def transcribe(self, audio: "np.ndarray") -> str:
        if not WHISPER_AVAILABLE:
            return ""
        loop = asyncio.get_event_loop()
        def _run():
            model = self._get_whisper()
            segments, _ = model.transcribe(audio, language="en", beam_size=1)
            return " ".join(s.text for s in segments).strip()
        return await loop.run_in_executor(None, _run)

    async def speak(self, text: str) -> None:
        if not TTS_AVAILABLE or not self._tts_enabled:
            return
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name

            communicate = edge_tts.Communicate(text, self.TTS_VOICE)
            await communicate.save(tmp_path)

            proc = await asyncio.create_subprocess_shell(
                f"mpg123 -q {tmp_path} 2>/dev/null || "
                f"ffplay -nodisp -autoexit -loglevel quiet {tmp_path} 2>/dev/null || "
                f"aplay {tmp_path} 2>/dev/null",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._tts_proc = proc
            await proc.wait()
            self._tts_proc = None
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    def stop_speaking(self) -> None:
        proc = self._tts_proc
        if proc and proc.returncode is None:
            try:
                proc.kill()
            except Exception:
                pass
        self._tts_proc = None

    def toggle_tts(self) -> bool:
        self._tts_enabled = not self._tts_enabled
        return self._tts_enabled


class LocalAgent:
    @staticmethod
    async def run_shell(command: str) -> str:
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
                parts.append(f"*(exited {proc.returncode} — no output)*")
            return "\n\n".join(parts)
        except asyncio.TimeoutError:
            return "❌ **Timeout** — command took > 30 s."
        except Exception as e:
            return f"❌ **Shell error:** {e}"

    @staticmethod
    async def read_file(path: str) -> str:
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return f"❌ Not found: `{p}`"
            if p.stat().st_size > 5 * 1024 * 1024:
                return f"❌ Too large (> 5 MB): `{p}`"
            content = p.read_text(errors="replace")
            lang = p.suffix.lstrip(".") or "text"
            return f"**File:** `{p}`\n\n```{lang}\n{content}\n```"
        except Exception as e:
            return f"❌ **Read error:** {e}"

    @staticmethod
    async def write_file(path: str, text: str) -> str:
        try:
            p = Path(path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            return f"✅ Written **{len(text):,}** chars → `{p}`"
        except Exception as e:
            return f"❌ **Write error:** {e}"

    @staticmethod
    async def fetch_url(url: str) -> str:
        if not WEB_AVAILABLE:
            return "❌ Install httpx + beautifulsoup4: `pip install httpx beautifulsoup4`"
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
                resp = await client.get(url, headers={"User-Agent": "ai-terminal-agent/2.1"})
                resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            if len(text) > 8000:
                text = text[:8000] + "\n\n*[truncated]*"
            return f"**URL:** {url}\n\n```\n{text}\n```"
        except Exception as e:
            return f"❌ **Fetch error:** {e}"

    @staticmethod
    async def open_app(app_name: str) -> str:
        app_name = app_name.strip()
        if not app_name:
            return "❌ No app name given."
        system = platform.system()
        try:
            if system == "Windows":
                subprocess.Popen(f'start "" "{app_name}"', shell=True)
            elif system == "Darwin":
                subprocess.Popen(["open", "-a", app_name])
            else:
                candidate = app_name.lower().replace(" ", "-")
                subprocess.Popen([candidate])
            return f"🚀 Opening **{app_name}**…"
        except Exception as e:
            return (
                f"❌ Couldn't open **{app_name}**: {e}\n"
                f"(On Linux, the app must be on your PATH under that exact name.)"
            )

    @staticmethod
    async def play_on_spotify(query: str) -> str:
        query = query.strip()
        if not query:
            return "❌ No song/artist given."
        try:
            opened = webbrowser.open(f"spotify:search:{quote(query)}")
            if not opened:
                webbrowser.open(f"https://open.spotify.com/search/{quote(query)}")
            return f"🎵 Opening Spotify and searching for **{query}**…"
        except Exception as e:
            return f"❌ Couldn't open Spotify: {e}"


_SPOTIFY_RE  = re.compile(r"^\s*play\s+(.+?)\s+on\s+spotify\s*$", re.IGNORECASE)
_OPEN_APP_RE = re.compile(r"^\s*open\s+(?:the\s+)?([A-Za-z0-9 _\-]{2,30}?)(?:\s+app)?\s*$", re.IGNORECASE)

_AGENT_CMD_LINE_RE = re.compile(r"^\s*(/shell|/read|/write|/fetch)\s+(.+)$", re.MULTILINE)


def match_local_action(text: str):
    text = text.strip()
    if not text or len(text.split()) > 8:
        return None
    m = _SPOTIFY_RE.match(text)
    if m:
        return ("spotify", m.group(1).strip())
    m = _OPEN_APP_RE.match(text)
    if m:
        target = m.group(1).strip()
        if len(target.split()) > 3:
            return None
        return ("open_app", target)
    return None


FALLBACK_MODELS = [
    ("OpenRouter Auto-Free Router", "openrouter/free"),
]


class ApiKeyScreen(Screen):
    CSS = """
    ApiKeyScreen {
        align: center middle;
        background: $surface;
    }
    #setup-card {
        width: 72;
        height: auto;
        border: thick $accent;
        padding: 2 3;
        background: $panel;
    }
    #setup-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    #step-indicator, #step-indicator-2 {
        color: $text-muted;
        padding-bottom: 1;
    }
    #step1-body, #step2-body {
        padding-bottom: 1;
    }
    #key-input {
        margin-bottom: 1;
    }
    #setup-error {
        color: $error;
        padding-bottom: 1;
    }
    #step1-buttons, #step2-buttons {
        height: 3;
        align-horizontal: right;
    }
    #step1-buttons Button, #step2-buttons Button {
        margin-left: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="setup-card"):
            yield Static("🔑  Welcome to ai-agent — one-time setup", id="setup-title")

            with Vertical(id="step-1"):
                yield Static("Step 1 of 2 — Get your OpenRouter API key", id="step-indicator")
                yield Static(
                    "ai-agent routes every prompt through OpenRouter, which needs "
                    "a free API key.\n\n"
                    "1. Create a free account at openrouter.ai\n"
                    "2. Open the Keys page and click 'Create Key'\n"
                    "3. Copy the key (starts with sk-or-...)\n\n"
                    f"{OPENROUTER_KEYS_URL}",
                    id="step1-body",
                )
                with Horizontal(id="step1-buttons"):
                    yield Button("Open openrouter.ai/keys", id="open-browser-btn")
                    yield Button("Next →", id="next-btn", variant="primary")

            with Vertical(id="step-2"):
                yield Static("Step 2 of 2 — Enter your API key", id="step-indicator-2")
                yield Static("Paste the key you copied from OpenRouter below.", id="step2-body")
                yield Input(placeholder="sk-or-...", password=True, id="key-input")
                yield Static("", id="setup-error")
                with Horizontal(id="step2-buttons"):
                    yield Button("← Back", id="back-btn")
                    yield Button("Save & continue", id="save-key-btn", variant="success")

    def on_mount(self) -> None:
        self.query_one("#step-2").display = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "open-browser-btn":
            try:
                webbrowser.open(OPENROUTER_KEYS_URL)
            except Exception:
                pass
        elif bid == "next-btn":
            self.query_one("#step-1").display = False
            self.query_one("#step-2").display = True
            self.query_one("#key-input", Input).focus()
        elif bid == "back-btn":
            self.query_one("#step-2").display = False
            self.query_one("#step-1").display = True
        elif bid == "save-key-btn":
            self._save_key()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "key-input":
            self._save_key()

    def _save_key(self) -> None:
        error = self.query_one("#setup-error", Static)
        key = self.query_one("#key-input", Input).value.strip()

        if not key:
            error.update("❌ Please paste a key first.")
            return
        if not key.startswith("sk-or-"):
            error.update(
                "⚠️  That doesn't look like a typical OpenRouter key "
                "(usually starts with sk-or-) — saving it anyway."
            )

        try:
            self._write_env_key(key)
        except Exception as e:
            error.update(f"❌ Could not save to .env: {e}")
            return

        os.environ["OPENROUTER_API_KEY"] = key
        self.app.pop_screen()

    @staticmethod
    def _write_env_key(key: str) -> None:
        env_path = Path(root_dir) / ".env"
        lines: list[str] = []
        found = False

        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    lines.append(f"OPENROUTER_API_KEY={key}")
                    found = True
                else:
                    lines.append(line)

        if not found:
            lines.append(f"OPENROUTER_API_KEY={key}")

        env_path.write_text("\n".join(lines) + "\n")

class ConfirmCommandScreen(ModalScreen[bool]):
    CSS = """
    ConfirmCommandScreen {
        align: center middle;
    }
    #confirm-card {
        width: 78;
        height: auto;
        border: thick $warning;
        padding: 2 3;
        background: $panel;
    }
    #confirm-title {
        text-style: bold;
        color: $warning;
        padding-bottom: 1;
    }
    #confirm-command {
        background: $surface;
        padding: 1;
        margin-bottom: 1;
    }
    #confirm-buttons {
        height: 3;
        align-horizontal: right;
    }
    #confirm-buttons Button {
        margin-left: 1;
    }
    """

    BINDINGS = [("escape", "skip", "Skip")]

    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-card"):
            yield Static("🤖  The AI wants to run this command:", id="confirm-title")
            yield Static(self.command, id="confirm-command")
            with Horizontal(id="confirm-buttons"):
                yield Button("❌ Skip", id="skip-btn")
                yield Button("✅ Run", id="run-btn", variant="success")

    def on_mount(self) -> None:
        self.query_one("#run-btn", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "run-btn")

    def action_skip(self) -> None:
        self.dismiss(False)


class AITerminalApp(App):

    BINDINGS = [
        ("ctrl+c",  "quit",         "Quit"),
        ("ctrl+l",  "clear",        "Clear Memory"),
        ("ctrl+p",  "toggle_panel", "Input Panel"),
        ("ctrl+y",  "copy_output",  "📋 Copy Reply"),
        ("f5",      "record_voice", "🎙 Record (STT)"),
        ("f6",      "toggle_tts",   "🔊 TTS on/off"),
        ("f7",      "toggle_agent", "🤖 Agent Mode"),
        ("f8",      "stop_tts",     "🛑 Stop TTS"),
    ]

    CSS = """
    Screen {
        background: $surface;
        layout: vertical;
    }

    #output-container {
        height: 1fr;
        border: solid $accent;
        padding: 1 2;
        margin: 1 1 0 1;
    }

    #voice-bar {
        height: 1;
        margin: 0 1;
        padding: 0 1;
        color: $text-muted;
    }
    .recording {
        color: $error;
        text-style: bold;
    }
    .tts-on {
        color: $success;
    }

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

    #input-area {
        height: auto;
        margin: 0 1 1 1;
    }
    #controls-row {
        height: 3;
        margin-bottom: 1;
    }
    #model-selector {
        width: 55%;
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
    """

    CONTENT_TYPES = [
        ("💬 Plain text / question",   "text"),
        ("🖼️  Image file path",         "image"),
        ("📄  Document / long text",    "document"),
        ("🗂️  File path → read & send", "filepath"),
        ("💻  Code snippet",            "code"),
    ]

    agent_mode: reactive[bool] = reactive(False)
    panel_open: reactive[bool] = reactive(False)
    is_recording: reactive[bool] = reactive(False)
    is_speaking:  reactive[bool] = reactive(False)
    tts_on: reactive[bool] = reactive(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat_history: list = [
            SystemMessage(content=(
                "You are a powerful CLI terminal assistant. "
                "You have access to local agent tools (shell, file R/W, web fetch). "
                "Keep responses concise and precise. Format code in fenced blocks."
            ))
        ]
        self.agent        = LocalAgent()
        self.voice        = VoiceEngine()
        self._models: list = FALLBACK_MODELS
        self.visible_transcript: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        yield VerticalScroll(
            Markdown(
                "### 🤖 AI Terminal  v2.1\n"
                "Models loading… please wait a moment.\n\n"
                "| Key | Action |\n"
                "|-----|--------|\n"
                "| `Ctrl+P` | Open / close input panel |\n"
                "| `F5` | Hold to record voice (STT) |\n"
                "| `F6` | Toggle TTS (speak responses) |\n"
                "| `F7` | Toggle Agent mode |\n"
                "| `Ctrl+L` | Clear memory |\n"
                "| `/shell <cmd>` | Run shell command |\n"
                "| `/read <path>` | Read a file |\n"
                "| `/write <path> <text>` | Write to file |\n"
                "| `/fetch <url>` | Fetch web page |",
                id="ai-response",
            ),
            id="output-container",
        )

        yield Static(
            "🎙 F5=Record  🔊 F6=TTS off  🤖 F7=Agent off  "
            + ("| Whisper ✓" if WHISPER_AVAILABLE else "| Whisper ✗ (pip install faster-whisper)")
            + ("  edge-tts ✓" if TTS_AVAILABLE else "  edge-tts ✗ (pip install edge-tts)"),
            id="voice-bar",
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
                "Paste code, text, image path, or file path — injected automatically on Enter.",
                id="panel-hint",
            )

        yield Container(
            Horizontal(
                Select(
                    FALLBACK_MODELS,
                    prompt="⏳ Loading models…",
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
        self.query_one("#input-panel").display = False
        self.load_models_async()
        self.preload_voice_model()

        if not os.environ.get("OPENROUTER_API_KEY"):
            self.push_screen(ApiKeyScreen())

    @work(thread=True)
    def preload_voice_model(self) -> None:
        if not WHISPER_AVAILABLE:
            return
        try:
            self.voice.preload_whisper()
        except Exception:
            pass

    @work
    async def load_models_async(self) -> None:
        selector = self.query_one("#model-selector", Select)
        status   = self.query_one("#status-label",   Static)

        if DYNAMIC_MODELS:
            try:
                models = await aget_available_models()
                self._models = models
                selector.set_options(models)
                if models:
                    selector.value = models[0][1]
                status.update(f"✅ {len(models)} free models loaded.")
            except Exception as e:
                status.update(f"⚠️  Model load failed: {e}")
                selector.set_options(FALLBACK_MODELS)
                selector.value = FALLBACK_MODELS[0][1]
        else:
            status.update("⚠️  open_router.py not found — using fallback.")
            selector.set_options(FALLBACK_MODELS)
            selector.value = FALLBACK_MODELS[0][1]

        self.query_one("#ai-response", Markdown).update(
            "### 🤖 AI Terminal  v2.1  — Ready\n\n"
            "| Key | Action |\n"
            "|-----|--------|\n"
            "| `Ctrl+P` | Open / close input panel |\n"
            "| `F5` | Press once to record voice (STT), press again to stop |\n"
            "| `F6` | Toggle TTS (speak AI responses aloud) |\n"
            "| `F7` | Toggle Agent mode |\n"
            "| `Ctrl+L` | Clear memory |\n"
            "| `/shell <cmd>` | Run shell command |\n"
            "| `/read <path>` | Read a file |\n"
            "| `/write <path> <text>` | Write to file |\n"
            "| `/fetch <url>` | Fetch web page |"
        )

    def action_toggle_panel(self) -> None:
        panel = self.query_one("#input-panel")
        self.panel_open = not self.panel_open
        panel.display = self.panel_open

    def action_toggle_agent(self) -> None:
        toggle = self.query_one("#agent-toggle", Switch)
        toggle.value = not toggle.value

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self.agent_mode = event.value
        voice_bar = self.query_one("#voice-bar", Static)
        self._update_voice_bar()
        status = self.query_one("#status-label", Static)
        if self.agent_mode:
            status.update("🤖 Agent ON — AI can suggest tool commands")
        else:
            status.update("Ready.")

    def action_record_voice(self) -> None:
        if not AUDIO_AVAILABLE:
            self.query_one("#status-label", Static).update(
                "❌ sounddevice not installed: pip install sounddevice numpy"
            )
            return
        if self.is_speaking:
            self.query_one("#status-label", Static).update(
                "🔊 Wait for the AI to finish speaking before recording…"
            )
            return
        if not self.is_recording:
            self._start_voice_recording()
        else:
            self._stop_voice_recording()

    def _start_voice_recording(self) -> None:
        if self.is_speaking:
            return
        if WHISPER_AVAILABLE and not self.voice.model_ready:
            self.query_one("#status-label", Static).update(
                "⏳ Voice model still loading (first run only) — try again in a few seconds…"
            )
            return
        ok = self.voice.start_recording()
        if ok:
            self.is_recording = True
            self._update_voice_bar()
            self.query_one("#status-label", Static).update("🔴 Recording… press F5 again to stop")

    def _stop_voice_recording(self) -> None:
        audio = self.voice.stop_recording()
        self.is_recording = False
        self._update_voice_bar()
        if audio is None or len(audio) == 0:
            self.query_one("#status-label", Static).update("⚠️  No audio captured.")
            return
        duration_s = len(audio) / self.voice.SAMPLE_RATE
        self.query_one("#status-label", Static).update(f"⏳ Transcribing {duration_s:.1f}s of audio…")
        self.transcribe_audio(audio)

    @work
    async def transcribe_audio(self, audio) -> None:
        status = self.query_one("#status-label", Static)
        prompt_input = self.query_one("#prompt-input", Input)
        try:
            text = await asyncio.wait_for(self.voice.transcribe(audio), timeout=25)
            if text:
                prompt_input.value = text
                status.update(f"🎙 Heard: \"{text}\" — sending…")
                prompt_input.value = ""
                await self._submit_prompt(text)
            else:
                status.update("⚠️  Transcription empty — try again.")
        except asyncio.TimeoutError:
            status.update("❌ Transcription timed out (> 25s) — try a shorter clip.")
        except Exception as e:
            status.update(f"❌ Transcription error: {e}")
            
    def action_stop_tts(self) -> None:
        self.voice.stop_speaking()
        self.is_speaking = False
        self._update_voice_bar()
        self.query_one("#status-label", Static).update("🛑 TTS stopped.")

    def action_toggle_tts(self) -> None:
        if not TTS_AVAILABLE:
            self.query_one("#status-label", Static).update(
                "❌ edge-tts not installed: pip install edge-tts"
            )
            return
        self.tts_on = self.voice.toggle_tts()
        self._update_voice_bar()
        self.query_one("#status-label", Static).update(
            f"🔊 TTS {'ON' if self.tts_on else 'OFF'}"
        )

    def _update_voice_bar(self) -> None:
        if self.is_speaking:
            rec_icon = "🗣️ Speaking…"
        elif self.is_recording:
            rec_icon = "🔴 Recording…"
        else:
            rec_icon = "🎙 F5=Record"
        tts_icon  = "🔊 TTS ON (conversational)" if self.tts_on else "🔊 F6=TTS off"
        agt_icon  = "🤖 Agent ON"   if self.agent_mode   else "🤖 F7=Agent off"
        self.query_one("#voice-bar", Static).update(
            f"{rec_icon}  {tts_icon}  {agt_icon}"
        )

    def _render_transcript(self) -> None:
        markdown = self.query_one("#ai-response", Markdown)
        if not self.visible_transcript:
            markdown.update(
                "### 🤖 AI Terminal  v2.1  — Ready\n"
                "Send a prompt to get started."
            )
            return
        markdown.update("\n\n---\n\n".join(self.visible_transcript))

    def _append_turn(self, block: str) -> None:
        self.visible_transcript.append(block)
        self._render_transcript()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_prompt = event.value.strip()
        if not user_prompt:
            return
        self.query_one("#prompt-input", Input).value = ""
        await self._submit_prompt(user_prompt)

    async def _submit_prompt(self, user_prompt: str) -> None:
        model_selector = self.query_one("#model-selector", Select)
        selected_model = model_selector.value
        if not selected_model or selected_model == Select.BLANK:
            self.query_one("#status-label", Static).update("⚠️  Select a model first!")
            return
        local_action = match_local_action(user_prompt)
        if local_action:
            self.run_local_action(user_prompt, local_action)
            return

        if user_prompt.startswith("/"):
            self.run_agent_command(user_prompt)
            return

        full_prompt = await self._build_prompt(user_prompt)

        display_name = str(selected_model).split("/")[-1]
        agent_tag    = " [Agent]" if self.agent_mode else ""
        self.query_one("#status-label", Static).update(
            f"⏳ {display_name}{agent_tag} thinking…"
        )

        self.run_llm_query(full_prompt, user_prompt, str(selected_model))

    async def _build_prompt(self, user_prompt: str) -> str:
        if not self.panel_open:
            return user_prompt

        panel_text   = self.query_one("#paste-area", TextArea).text.strip()
        content_type = self.query_one("#content-type-selector", Select).value or "text"

        if not panel_text:
            return user_prompt

        if content_type == "filepath":
            file_result = await self.agent.read_file(panel_text)
            return f"{user_prompt}\n\n---\n{file_result}"

        if content_type == "image":
            p = Path(panel_text).expanduser()
            if p.exists():
                try:
                    b64 = base64.b64encode(p.read_bytes()).decode()
                    return (
                        f"{user_prompt}\n\n"
                        f"[Image: `{p.name}` base64 len={len(b64)}. Analyse as requested.]"
                    )
                except Exception:
                    pass
            return f"{user_prompt}\n\n[Image path: `{panel_text}` — not readable]"

        if content_type == "code":
            lang = Path(panel_text.splitlines()[0]).suffix.lstrip(".") if panel_text else ""
            return f"{user_prompt}\n\n```{lang}\n{panel_text}\n```"

        if content_type == "document":
            return f"{user_prompt}\n\n---\n**Document:**\n{panel_text}"

        return f"{user_prompt}\n\n---\n{panel_text}"

    @work(exclusive=False)
    async def run_local_action(self, display_prompt: str, action) -> None:
        kind, target = action
        status = self.query_one("#status-label", Static)

        if kind == "spotify":
            status.update(f"🎵 Opening Spotify: {target}…")
            result = await self.agent.play_on_spotify(target)
        else:
            status.update(f"🚀 Opening {target}…")
            result = await self.agent.open_app(target)

        self._append_turn(f"### 🗣️ You:\n{display_prompt}\n\n{result}")
        status.update("✅ Done.")

        if self.tts_on:
            self.speak_response(result)

    async def _execute_agent_command(self, command: str) -> str:
        parts = command.split(maxsplit=2)
        verb  = parts[0].lower()

        if verb == "/shell":
            cmd = " ".join(parts[1:]) if len(parts) > 1 else ""
            return await self.agent.run_shell(cmd) if cmd else "❌ Usage: `/shell <command>`"
        elif verb == "/read":
            path = parts[1] if len(parts) > 1 else ""
            return await self.agent.read_file(path) if path else "❌ Usage: `/read <path>`"
        elif verb == "/write":
            return (await self.agent.write_file(parts[1], parts[2])
                     if len(parts) >= 3 else "❌ Usage: `/write <path> <text>`")
        elif verb == "/fetch":
            url = parts[1] if len(parts) > 1 else ""
            return await self.agent.fetch_url(url) if url else "❌ Usage: `/fetch <url>`"
        else:
            return f"❌ Unknown: `{verb}`. Try `/shell` `/read` `/write` `/fetch`"

    @work(exclusive=False)
    async def run_agent_command(self, command: str) -> None:
        status = self.query_one("#status-label", Static)
        verb   = command.split(maxsplit=1)[0].lower()
        status.update(f"⚙️  Running `{verb}`…")

        result = await self._execute_agent_command(command)

        self._append_turn(f"### ⚙️ Agent: `{command}`\n\n{result}")
        status.update("✅ Done.")

    @work(exclusive=False)
    async def auto_execute_suggested_commands(self, reply_text: str) -> None:
        matches = _AGENT_CMD_LINE_RE.findall(reply_text)
        if not matches:
            return
        status = self.query_one("#status-label", Static)
        for verb, rest in matches[:5]:
            command_line = f"{verb.lower()} {rest}".strip()

            confirmed = await self.push_screen_wait(ConfirmCommandScreen(command_line))
            if not confirmed:
                self._append_turn(f"### ⚙️ Skipped: `{command_line}`\n\n*(declined — not run)*")
                continue

            status.update(f"⚙️ Running `{command_line}`…")
            result = await self._execute_agent_command(command_line)
            self._append_turn(f"### ⚙️ Auto-executed: `{command_line}`\n\n{result}")
        status.update("✅ Done.")

    @work(exclusive=True)
    async def run_llm_query(self, full_prompt: str, display_prompt: str, model_id: str) -> None:
        status = self.query_one("#status-label", Static)

        self._append_turn(f"### 💬 You:\n{display_prompt}\n\n*Thinking…*")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            self.visible_transcript[-1] = (
                f"### 💬 You:\n{display_prompt}\n\n"
                "❌ **Error:** `OPENROUTER_API_KEY` not set in `.env`."
            )
            self._render_transcript()
            status.update("❌ Missing API Key.")
            return

        try:
            llm = ChatOpenRouter(model=model_id, api_key=api_key, temperature=0.2)

            content = full_prompt
            if self.agent_mode:
                content = (
                    "[AGENT MODE] Suggest tool commands when helpful:\n"
                    "  /shell <cmd>  /read <path>  /write <path> <text>  /fetch <url>\n\n"
                ) + content

            self.chat_history.append(HumanMessage(content=content))
            response = await llm.ainvoke(self.chat_history)
            self.chat_history.append(response)

            reply_text = response.content
            self.visible_transcript[-1] = f"### 💬 You:\n{display_prompt}\n\n{reply_text}"
            self._render_transcript()
            status.update("✅ Done.")

            if self.agent_mode:
                self.auto_execute_suggested_commands(reply_text)

            if self.tts_on:
                import re
                plain = re.sub(r"[`#*_\[\]()]", "", reply_text)
                self.speak_response(plain[:1000]) 

        except Exception as e:
            self.visible_transcript[-1] = (
                f"### 💬 You:\n{display_prompt}\n\n❌ **Error:**\n```\n{e}\n```"
            )
            self._render_transcript()
            status.update("❌ Error.")

    @work(exclusive=False)
    async def speak_response(self, text: str) -> None:
        self.is_speaking = True
        self._update_voice_bar()
        try:
            await self.voice.speak(text)
        finally:
            self.is_speaking = False
            self._update_voice_bar()
            if (
                self.tts_on
                and AUDIO_AVAILABLE
                and not self.is_recording
                and WHISPER_AVAILABLE
            ):
                self._start_voice_recording()

    def action_copy_output(self) -> None:
        status = self.query_one("#status-label", Static)
        if not self.visible_transcript:
            status.update("⚠️  Nothing to copy yet.")
            return

        last_turn = self.visible_transcript[-1]
        if "\n\n" in last_turn:
            _, _, reply = last_turn.partition("\n\n")
        else:
            reply = last_turn

        self.copy_to_clipboard(reply.strip())
        status.update("📋 Copied last reply to clipboard.")

    def action_clear(self) -> None:
        self.chat_history = [
            SystemMessage(content=(
                "You are a powerful CLI terminal assistant with local agent tools. "
                "Keep responses concise and precise."
            ))
        ]
        self.visible_transcript = []
        self.query_one("#ai-response", Markdown).update(
            "### 🧹 Memory cleared. Ready."
        )
        self.query_one("#status-label", Static).update("Ready.")
        try:
            self.query_one("#paste-area", TextArea).load_text("")
        except Exception:
            pass


def main():
    app = AITerminalApp()
    app.run()

if __name__ == "__main__":
    main()