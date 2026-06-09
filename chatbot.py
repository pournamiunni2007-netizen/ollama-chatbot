import streamlit as st
import ollama
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ollama Chat",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS  ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }

    /* warm off-white background */
    .stApp { background: #faf8f5; }

    /* sidebar warm parchment */
    section[data-testid="stSidebar"] {
        background: #f0ebe3;
        border-right: 1px solid #ddd5c8;
    }
    section[data-testid="stSidebar"] * { color: #3b2f2f !important; }

    /* chat bubbles */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #e8e0d8 !important;
        border-radius: 14px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* user bubble — soft amber tint */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #fff7ed !important;
        border-color: #f5c97a !important;
    }

    /* title */
    h3 { color: #7c3d12 !important; letter-spacing: -0.3px; }

    /* input box */
    .stChatInputContainer textarea {
        background: #ffffff !important;
        border: 1.5px solid #ddd5c8 !important;
        border-radius: 12px !important;
        color: #3b2f2f !important;
    }
    .stChatInputContainer textarea:focus {
        border-color: #c97f3b !important;
        box-shadow: 0 0 0 3px rgba(201,127,59,0.15) !important;
    }

    /* buttons */
    .stButton > button {
        background: #c97f3b !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background: #a8622a !important;
    }

    /* selectbox */
    [data-baseweb="select"] {
        border-color: #ddd5c8 !important;
    }

    .badge-online  { color: #16a34a; font-size: 13px; font-weight: 600; }
    .badge-offline { color: #dc2626; font-size: 13px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def check_ollama() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False


def get_models() -> list:
    """Return list of installed model names using the correct Ollama SDK API."""
    try:
        result = ollama.list()           # returns ListResponse object
        return [m.model for m in result.models if m.model]
    except Exception:
        return []


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful, concise assistant."


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💬 Ollama Chat")
    st.markdown("---")

    connected = check_ollama()

    if connected:
        st.markdown('<p class="badge-online">● Connected to Ollama</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<p class="badge-offline">● Ollama not running</p>',
                    unsafe_allow_html=True)
        st.error("Start Ollama first:\n```\nollama serve\n```\nThen refresh this page.")

    st.markdown("---")

    models = get_models()

    if models:
        selected_model = st.selectbox("🧠 Model", models, index=0)
    else:
        st.warning("No models found.\n\nRun:\n```\nollama pull llama3\n```")
        selected_model = None

    st.markdown("---")

    system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.system_prompt,
        height=110,
    )

    if system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        st.caption(f"{len(st.session_state.messages)} messages")


# ── Main chat area ─────────────────────────────────────────────────────────────
st.markdown("### 💬 Chat")

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:3.5rem 0;color:#a8916e;">
        <div style="font-size:3.5rem;">🤖</div>
        <p style="font-size:1.15rem;font-weight:600;color:#7c4f2a;margin-top:0.6rem;">
            Ask me anything
        </p>
        <p style="font-size:0.875rem;color:#b09070;">
            Running privately on your machine via Ollama.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ──────────────────────────────────────────────────────────────────────
prompt = st.chat_input(
    "Type a message...",
    disabled=(not connected or selected_model is None),
)

if prompt:
    if not connected:
        st.error("Ollama is not running. Open a terminal and run `ollama serve`.")
        st.stop()

    if not selected_model:
        st.error("No model found. Run `ollama pull llama3` in your terminal.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build message list with system prompt prepended
    api_messages = [{"role": "system", "content": st.session_state.system_prompt}]
    for m in st.session_state.messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    # Stream response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            stream = ollama.chat(
                model=selected_model,
                messages=api_messages,
                stream=True,
            )

            for chunk in stream:
                # ChatResponse object → chunk.message.content
                token = chunk.message.content or ""
                full_response += token
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        except ollama.ResponseError as e:
            full_response = f"⚠️ Model error: {e.error}"
            placeholder.error(full_response)

        except ConnectionError:
            full_response = "⚠️ Lost connection to Ollama. Is `ollama serve` still running?"
            placeholder.error(full_response)

        except Exception as e:
            full_response = f"⚠️ Error: {str(e)}"
            placeholder.error(full_response)

    if full_response:
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
