import sqlite3
import gradio as gr
from openai import OpenAI
from typing import Generator

# --- Database Setup ---

DB_PATH = "chat_history.db"

def init_db() -> None:
    """Initialize the SQLite database and create messages table if it doesn't exist."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
    except sqlite3.Error as e:
        print(f"DB init error: {e}")

def save_to_db(role: str, content: str) -> None:
    """Save a message to the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO messages (role, content) VALUES (?, ?)",
                (role, content)
            )
    except sqlite3.Error as e:
        print(f"DB save error: {e}")

def load_history_from_db() -> list[dict]:
    """Load chat history from the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT role, content FROM messages ORDER BY id ASC")
            return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"DB load error: {e}")
        return []

def load_settings_from_db(defaults: dict) -> dict:
    """Load settings from DB, falling back to provided defaults."""
    settings = defaults.copy()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT key, value FROM settings")
            for key, value in cursor.fetchall():
                if key in settings:
                    # Cast numeric fields back to float/int where appropriate
                    if key in {"temperature"}:
                        settings[key] = float(value)
                    elif key in {"max_tokens"}:
                        settings[key] = int(value)
                    else:
                        settings[key] = value
    except sqlite3.Error as e:
        print(f"DB settings load error: {e}")
    return settings

def save_settings_to_db(settings: dict) -> None:
    """Persist settings dictionary to DB."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.executemany(
                "REPLACE INTO settings (key, value) VALUES (?, ?)",
                [(k, str(v)) for k, v in settings.items()]
            )
    except sqlite3.Error as e:
        print(f"DB settings save error: {e}")

# --- Logic Functions ---
def update_mentions(history: list[dict]) -> dict:
    """
    Updates dropdown with choices from history.
    Args:
        history: List of message dicts.
    Returns:
        gr.Update object for dropdown choices.
    """
    if not history:
        return gr.update(choices=["None"], value="None")
    options = [f"[{i}] {msg['content'][:50]}..." for i, msg in enumerate(history)]
    return gr.update(choices=["None"] + options)


def persist_settings(system_prompt: str, temperature: float, max_tokens: int, ref_choice: str,
                     api_base: str, api_key: str, model_name: str):
    """Persist user-configurable settings to the database."""
    save_settings_to_db({
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "ref_choice": ref_choice,
        "api_base": api_base,
        "api_key": api_key,
        "model_name": model_name,
    })
    # No UI update needed
    return None

def chat_func(
        message: str,
        history: list,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        ref_choice: str,
        api_base: str,
        api_key: str,
        model_name: str
    ) -> Generator[str, None, None]:
        """
        Handles chat interaction, streaming LLM response, and saving messages to DB.
        """
        try:
            client = OpenAI(base_url=api_base, api_key=api_key)
        except Exception as e:
            yield f"[Error initializing OpenAI client: {e}]"
            return

        # Build messages list
        messages = [{"role": "system", "content": system_prompt}]

        # Add Mention Context (if selected)
        if ref_choice != "None":
            try:
                ref_idx = int(ref_choice.split(']')[0][1:])
                ref_msg = history[ref_idx]
                messages.append({"role": "system", "content": f"Reference context: {ref_msg['content']}"})
            except Exception as e:
                yield f"[Reference context error: {e}]"

        # Add History
        messages.extend(history)

        # Add current user message and save to DB
        messages.append({"role": "user", "content": message})
        save_to_db("user", message)

        # Get LLM response
        full_reply = ""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in response:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    full_reply += delta
                    yield full_reply
        except Exception as e:
            yield f"[OpenAI API error: {e}]"
            return

        # Save assistant reply to DB after completion
        save_to_db("assistant", full_reply)

# --- UI Layout ---
init_db()
initial_history = load_history_from_db()
default_settings = {
    "system_prompt": "Be concise.",
    "temperature": 0.7,
    "max_tokens": 512,
    "ref_choice": "None",
    "api_base": "api.openai.com",
    "api_key": "",
    "model_name": "gpt-4o",
}
persisted_settings = load_settings_from_db(default_settings)

with gr.Blocks(title="AI传声筒 Senior Editor") as demo:

    gr.Markdown("""
    <div style='display: flex; align-items: center; gap: 12px;'>
        <img src='https://img.icons8.com/color/48/000000/sql.png' style='height:40px;'>
        <span style='font-size:2rem;font-weight:600;'>Persistent LLM Chat</span>
    </div>
    <hr>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Group():
                system_prompt = gr.Textbox(label="System Prompt", value=persisted_settings["system_prompt"], lines=2)
                temperature = gr.Slider(0, 2, persisted_settings["temperature"], label="Temperature", info="Controls randomness")
                max_tokens = gr.Slider(64, 100024, persisted_settings["max_tokens"], step=64, label="Max Tokens", info="Max response length")
                mention_dropdown = gr.Dropdown(label="Mention Specific Response", choices=["None"], value=persisted_settings["ref_choice"])
                api_base = gr.Textbox(label="API Base", value=persisted_settings["api_base"])
                api_key = gr.Textbox(label="API Key", value=persisted_settings["api_key"], type="password")
                model_name = gr.Textbox(label="Model", value=persisted_settings["model_name"])

        with gr.Column(scale=5):
            chatbot = gr.Chatbot(
                value=initial_history,
                placeholder="今日主题: ",
            )

            chat_interface = gr.ChatInterface(
                fn=chat_func,
                chatbot=chatbot,
                additional_inputs=[
                    system_prompt,
                    temperature,
                    max_tokens,
                    mention_dropdown,
                    api_base,
                    api_key,
                    model_name
                ]
            )

    # Efficiently update the mention dropdown when history changes
    chatbot.change(update_mentions, inputs=[chatbot], outputs=[mention_dropdown])

    # Persist settings automatically on change
    for comp in [system_prompt, temperature, max_tokens, mention_dropdown, api_base, api_key, model_name]:
        comp.change(
            persist_settings,
            inputs=[system_prompt, temperature, max_tokens, mention_dropdown, api_base, api_key, model_name],
            outputs=[]
        )

    # Custom CSS for further beautification
    demo.css = """
    #chatbot-box {
        background: #f8fafc;
        border-radius: 16px;
        box-shadow: 0 2px 8px #0001;
        padding: 12px;
    }
    .gradio-container {
        background: linear-gradient(120deg, #e0e7ff 0%, #f8fafc 100%);
    }
    """

if __name__ == "__main__":
    demo.launch()
 
