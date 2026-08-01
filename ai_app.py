import os
import asyncio
from dotenv import load_dotenv
load_dotenv()
from textual.app import App, ComposeResult
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Container, Horizontal
from textual.widgets import Header, Footer, Input, Markdown, Static, Select
from textual import work

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage

class AITerminalApp(App):
    CSS = """
    Screen {
        background: $surface;
        layout: vertical;
    }
    #output-container {
        height: 1fr;
        border: solid $accent;
        padding: 1 2;
        margin: 1;
    }
    #input-area {
        height: auto;
        margin: 0 1 1 1;
    }
    #controls-row {
        height: auto;
        margin-bottom: 1;
    }
    #model-selector {
        width: 50%;
    }
    #prompt-input {
        border: tall $primary;
    }
    .status-msg {
        color: $text-muted;
        padding-left: 2;
        padding-top: 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear", "Clear Screen"),
    ]

    AVAILABLE_MODELS = [
        ("Cohere North Mini Code (256K Context)", "cohere/north-mini-code:free"),
        ("NVIDIA Nemotron 3 Super (Free)", "nvidia/nemotron-3-super-120b-a12b:free"),
        ("NVIDIA Nemotron 3 Ultra (1M Context)", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        ("Google Gemma 4 31B (Vision & Text)", "google/gemma-4-31b-it:free"),
        ("OpenRouter Auto-Free Router", "openrouter/free")
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize conversation memory
        self.chat_history = [
            SystemMessage(content="You are a CLI terminal assistant. Keep all responses concise, direct, and focused.")
        ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        yield VerticalScroll(
            Markdown("### 🦜🔗 LangChain AI Terminal\nSelect a model, type your prompt, and press **Enter**.", id="ai-response"),
            id="output-container"
        )

        yield Container(
            Horizontal(
                Select(
                    self.AVAILABLE_MODELS,
                    prompt="Select a Model",
                    value=self.AVAILABLE_MODELS[0][1],
                    id="model-selector"
                ),
                Static("Ready.", id="status-label", classes="status-msg"),
                id="controls-row"
            ),
            Input(placeholder="Ask your AI agent something...", id="prompt-input"),
            id="input-area"
        )

        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_prompt = event.value.strip()
        if not user_prompt:
            return

        model_selector = self.query_one("#model-selector", Select)
        selected_model = model_selector.value

        if not selected_model:
            status_label = self.query_one("#status-label", Static)
            status_label.update("⚠️ Please select a model first!")
            return

        input_widget = self.query_one("#prompt-input", Input)
        input_widget.value = ""

        status_label = self.query_one("#status-label", Static)
        display_name = selected_model.split('/')[-1] if '/' in selected_model else selected_model
        status_label.update(f"⏳ LangChain invoking {display_name}...")

        self.run_llm_query(user_prompt, selected_model)

    @work(exclusive=True)
    async def run_llm_query(self, prompt: str, model_id: str) -> None:
        markdown_widget = self.query_one("#ai-response", Markdown)
        status_label = self.query_one("#status-label", Static)

        markdown_widget.update(f"### 💬 Prompt:\n{prompt}\n\n---\n\n*Thinking with LangChain...*")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            markdown_widget.update("❌ **Error:** `OPENROUTER_API_KEY` environment variable is not set.")
            status_label.update("❌ Missing API Key.")
            return

        try:
            llm = ChatOpenRouter(
                model=model_id,
                api_key=api_key,
                temperature=0.2
            )

            # 1. Add your new prompt to the memory
            self.chat_history.append(HumanMessage(content=prompt))

            # 2. Send the ENTIRE memory to the model
            response = await llm.ainvoke(self.chat_history)
            
            # 3. Add the model's response back to the memory
            self.chat_history.append(response)
            
            markdown_widget.update(f"### 💬 Prompt:\n{prompt}\n\n---\n\n{response.content}")
            status_label.update("✅ Done!")

        except Exception as e:
            markdown_widget.update(f"**LangChain Execution Error:**\n```\n{str(e)}\n```")
            status_label.update("❌ Error occurred.")

    def action_clear(self) -> None:
        # Wipe the memory when screen is cleared
        self.chat_history = [
            SystemMessage(content="You are a CLI terminal assistant. Keep all responses concise, direct, and focused.")
        ]
        
        markdown_widget = self.query_one("#ai-response", Markdown)
        markdown_widget.update("### Output and memory cleared.\nReady for your next prompt.")
        status_label = self.query_one("#status-label", Static)
        status_label.update("Ready.")

if __name__ == "__main__":
    app = AITerminalApp()
    app.run()
