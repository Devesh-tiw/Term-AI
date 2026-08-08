"""
ai-terminal-app  ·  __main__.py
Author : Devesh Tiwari — AI Engineering, Amrita Vishwa Vidyapeetham
Version: 2.1.0

Fixes in this version
──────────────────────
1. BUG FIX  — Wrong import name (open_router → openrouter_models)
2. BUG FIX  — App hang on startup: AVAILABLE_MODELS was fetched at class
              definition time (blocking). Now loaded async in on_mount via @work.
3. BUG FIX  — Ctrl+A intercepted by terminal before Textual. Changed to F5/F6/F7.
4. FEATURE  — Voice Engine: F5 = record (Whisper STT), F6 = toggle TTS (edge-tts)

Keybindings (updated)
──────────────────────
  Enter    → submit prompt
  Ctrl+L   → clear memory
  Ctrl+P   → toggle rich input panel
  Ctrl+C   → quit
  F5       → record voice (hold → speak → release to transcribe)
  F6       → toggle TTS on/off
  F7       → toggle Agent mode  (was Ctrl+A — terminal conflict fixed)
"""

import os
import asyncio
import base64
import tempfile
import threading
from pathlib import Path

from dotenv import load_dotenv

# ── env loading ──────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.dirname(current_dir)
load_dotenv(os.path.join(root_dir, ".env"))

# ── Textual ──────────────────────────────────────────────────────────────────
from textual.app        import App, ComposeResult
from textual.containers import VerticalScroll, Container, Horizontal, Vertical
from textual.widgets    import (
    Header, Footer, Input, Markdown, Static, Select,
    TextArea, Switch,
)
from textual            import work
from textual.reactive   import reactive

# ── LangChain ────────────────────────────────────────────────────────────────
from langchain_openrouter    import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage

# ── Optional deps (graceful degradation if not installed) ───────────────────
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

# ── Model loader (FIX: was 'open_router', correct name is 'openrouter_models') ──
try:
    from openrouter_models import get_available_models, aget_available_models
    DYNAMIC_MODELS = True
except ImportError:
    DYNAMIC_MODELS = False


# ═══════════════════════════════════════════════════════════════════════════════
#  VOICE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceEngine:
    """
    STT: faster-whisper (local, offline, tiny model ~75MB).
    TTS: edge-tts (Microsoft Azure voices, free, async, excellent quality).

    Install:
        pip install faster-whisper sounddevice numpy edge-tts
    """

    WHISPER_MODEL_SIZE = "tiny"   # tiny/base/small/medium — tiny is fastest
    SAMPLE_RATE        = 16_000   # Whisper expects 16 kHz
    TTS_VOICE          = "en-US-AriaNeural"  # change to taste

    def __init__(self):
        self._whisper: "WhisperModel | None" = None
        self._recording       = False
        self._recorded_frames: list = []
        self._tts_enabled     = False
        self._record_lock     = threading.Lock()

    # ── Lazy-load Whisper (first STT call only) ───────────────────────────
    def _get_whisper(self):
        if self._whisper is None:
            if not WHISPER_AVAILABLE:
                raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")
            self._whisper = WhisperModel(
                self.WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
        return self._whisper

    # ── Record until stop_recording() is called ───────────────────────────
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

    # ── Transcribe audio array → text ─────────────────────────────────────
    async def transcribe(self, audio: "np.ndarray") -> str:
        if not WHISPER_AVAILABLE:
            return ""
        loop = asyncio.get_event_loop()
        def _run():
            model = self._get_whisper()
            segments, _ = model.transcribe(audio, language="en", beam_size=1)
            return " ".join(s.text for s in segments).strip()
        return await loop.run_in_executor(None, _run)

    # ── Speak text via edge-tts ───────────────────────────────────────────
    async def speak(self, text: str) -> None:
        if not TTS_AVAILABLE or not self._tts_enabled:
            return
        try:
            # edge-tts streams MP3; we save to tmp and play with sounddevice
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name

            communicate = edge_tts.Communicate(text, self.TTS_VOICE)
            await communicate.save(tmp_path)

            # Play with an OS-level player (no extra deps)
            proc = await asyncio.create_subprocess_shell(
                f"mpg123 -q {tmp_path} 2>/dev/null || "
                f"ffplay -nodisp -autoexit -loglevel quiet {tmp_path} 2>/dev/null || "
                f"aplay {tmp_path} 2>/dev/null",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass   # TTS is best-effort; never crash the main app

    def toggle_tts(self) -> bool:
        self._tts_enabled = not self._tts_enabled
        return self._tts_enabled


# ═══════════════════════════════════════════════════════════════════════════════
#  LOCAL AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class LocalAgent:
    """
    Slash commands:
        /shell <cmd>           – run a shell command
        /read  <path>          – read a file from disk
        /write <path> <text>   – write text to a file
        /fetch <url>           – fetch a URL and return cleaned text
    """

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


# ═══════════════════════════════════════════════════════════════════════════════
#  EMERGENCY FALLBACK MODELS (used if openrouter_models.py import fails)
# ═══════════════════════════════════════════════════════════════════════════════
FALLBACK_MODELS = [
    ("OpenRouter Auto-Free Router", "openrouter/free"),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

class AITerminalApp(App):

    # ── Keybindings ──────────────────────────────────────────────────────────
    # FIX: Ctrl+A is grabbed by the terminal (select-all) before Textual sees it.
    #      Moved agent toggle to F7. Voice uses F5/F6 (universally safe).
    BINDINGS = [
        ("ctrl+c",  "quit",         "Quit"),
        ("ctrl+l",  "clear",        "Clear Memory"),
        ("ctrl+p",  "toggle_panel", "Input Panel"),
        ("f5",      "record_voice", "🎙 Record (STT)"),
        ("f6",      "toggle_tts",   "🔊 TTS on/off"),
        ("f7",      "toggle_agent", "🤖 Agent Mode"),
    ]

    CSS = """
    Screen {
        background: $surface;
        layout: vertical;
    }

    /* ── Chat output ────────────────────────── */
    #output-container {
        height: 1fr;
        border: solid $accent;
        padding: 1 2;
        margin: 1 1 0 1;
    }

    /* ── Voice status bar ───────────────────── */
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

    /* ── Rich input panel ───────────────────── */
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

    /* ── Bottom bar ─────────────────────────── */
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

    # ── Reactive state ────────────────────────────────────────────────────
    agent_mode: reactive[bool] = reactive(False)
    panel_open: reactive[bool] = reactive(False)
    is_recording: reactive[bool] = reactive(False)
    tts_on: reactive[bool] = reactive(False)

    # ── Init ──────────────────────────────────────────────────────────────
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
        # FIX: Do NOT call get_available_models() here — it blocks.
        #      Models are loaded async in on_mount via @work.
        self._models: list = FALLBACK_MODELS

    # ── Layout ────────────────────────────────────────────────────────────
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

        # Voice status bar
        yield Static(
            "🎙 F5=Record  🔊 F6=TTS off  🤖 F7=Agent off  "
            + ("| Whisper ✓" if WHISPER_AVAILABLE else "| Whisper ✗ (pip install faster-whisper)")
            + ("  edge-tts ✓" if TTS_AVAILABLE else "  edge-tts ✗ (pip install edge-tts)"),
            id="voice-bar",
        )

        # Rich input panel (hidden by default)
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

        # Bottom: model select + agent toggle + status + prompt
        yield Container(
            Horizontal(
                Select(
                    FALLBACK_MODELS,     # placeholder; replaced after models load
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

    # ── Mount: load models async so UI opens instantly ────────────────────
    def on_mount(self) -> None:
        self.query_one("#input-panel").display = False
        self.load_models_async()

    @work
    async def load_models_async(self) -> None:
        """Fetch the live free-model list without blocking the UI."""
        selector = self.query_one("#model-selector", Select)
        status   = self.query_one("#status-label",   Static)

        if DYNAMIC_MODELS:
            try:
                models = await aget_available_models()
                self._models = models
                # Rebuild the Select widget with real model list
                selector.set_options(models)
                if models:
                    selector.value = models[0][1]
                status.update(f"✅ {len(models)} free models loaded.")
            except Exception as e:
                status.update(f"⚠️  Model load failed: {e}")
                selector.set_options(FALLBACK_MODELS)
                selector.value = FALLBACK_MODELS[0][1]
        else:
            status.update("⚠️  openrouter_models.py not found — using fallback.")
            selector.set_options(FALLBACK_MODELS)
            selector.value = FALLBACK_MODELS[0][1]

        # Update welcome text now that models are loaded
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

    # ── Panel toggle ──────────────────────────────────────────────────────
    def action_toggle_panel(self) -> None:
        panel = self.query_one("#input-panel")
        self.panel_open = not self.panel_open
        panel.display = self.panel_open

    # ── Agent toggle (F7 — was Ctrl+A, fixed terminal conflict) ──────────
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

    # ── Voice: F5 = toggle record ─────────────────────────────────────────
    def action_record_voice(self) -> None:
        if not AUDIO_AVAILABLE:
            self.query_one("#status-label", Static).update(
                "❌ sounddevice not installed: pip install sounddevice numpy"
            )
            return
        if not self.is_recording:
            self._start_voice_recording()
        else:
            self._stop_voice_recording()

    def _start_voice_recording(self) -> None:
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
        self.query_one("#status-label", Static).update("⏳ Transcribing…")
        self.transcribe_audio(audio)

    @work
    async def transcribe_audio(self, audio) -> None:
        status = self.query_one("#status-label", Static)
        prompt_input = self.query_one("#prompt-input", Input)
        try:
            text = await self.voice.transcribe(audio)
            if text:
                prompt_input.value = text
                status.update(f"🎙 Transcribed — press Enter to send")
            else:
                status.update("⚠️  Transcription empty — try again.")
        except Exception as e:
            status.update(f"❌ Transcription error: {e}")

    # ── Voice: F6 = toggle TTS ────────────────────────────────────────────
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
        rec_icon  = "🔴 Recording…" if self.is_recording else "🎙 F5=Record"
        tts_icon  = "🔊 TTS ON"     if self.tts_on       else "🔊 F6=TTS off"
        agt_icon  = "🤖 Agent ON"   if self.agent_mode   else "🤖 F7=Agent off"
        self.query_one("#voice-bar", Static).update(
            f"{rec_icon}  {tts_icon}  {agt_icon}"
        )

    # ── Submit ────────────────────────────────────────────────────────────
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_prompt = event.value.strip()
        if not user_prompt:
            return

        model_selector = self.query_one("#model-selector", Select)
        selected_model = model_selector.value
        if not selected_model or selected_model == Select.BLANK:
            self.query_one("#status-label", Static).update("⚠️  Select a model first!")
            return

        self.query_one("#prompt-input", Input).value = ""

        # Slash command → agent tool
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

    # ── Build enriched prompt from panel ─────────────────────────────────
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

    # ── Agent slash commands ──────────────────────────────────────────────
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
            result = (await self.agent.write_file(parts[1], parts[2])
                      if len(parts) >= 3 else "❌ Usage: `/write <path> <text>`")
        elif verb == "/fetch":
            url    = parts[1] if len(parts) > 1 else ""
            result = await self.agent.fetch_url(url) if url else "❌ Usage: `/fetch <url>`"
        else:
            result = f"❌ Unknown: `{verb}`. Try `/shell` `/read` `/write` `/fetch`"

        markdown.update(f"### ⚙️ Agent: `{command}`\n\n---\n\n{result}")
        status.update("✅ Done.")

    # ── LLM query ─────────────────────────────────────────────────────────
    @work(exclusive=True)
    async def run_llm_query(self, full_prompt: str, display_prompt: str, model_id: str) -> None:
        markdown = self.query_one("#ai-response", Markdown)
        status   = self.query_one("#status-label", Static)

        markdown.update(f"### 💬 You:\n{display_prompt}\n\n---\n\n*Thinking…*")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            markdown.update("❌ **Error:** `OPENROUTER_API_KEY` not set in `.env`.")
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
            markdown.update(f"### 💬 You:\n{display_prompt}\n\n---\n\n{reply_text}")
            status.update("✅ Done.")

            # TTS: speak the response if enabled
            if self.tts_on:
                # Strip markdown for cleaner speech
                import re
                plain = re.sub(r"[`#*_\[\]()]", "", reply_text)
                self.speak_response(plain[:1000])   # cap at 1000 chars

        except Exception as e:
            markdown.update(
                f"### 💬 You:\n{display_prompt}\n\n---\n\n❌ **Error:**\n```\n{e}\n```"
            )
            status.update("❌ Error.")

    @work(exclusive=False)
    async def speak_response(self, text: str) -> None:
        await self.voice.speak(text)

    # ── Clear ─────────────────────────────────────────────────────────────
    def action_clear(self) -> None:
        self.chat_history = [
            SystemMessage(content=(
                "You are a powerful CLI terminal assistant with local agent tools. "
                "Keep responses concise and precise."
            ))
        ]
        self.query_one("#ai-response", Markdown).update(
            "### 🧹 Memory cleared. Ready."
        )
        self.query_one("#status-label", Static).update("Ready.")
        try:
            self.query_one("#paste-area", TextArea).load_text("")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    app = AITerminalApp()
    app.run()

if __name__ == "__main__":
    main()