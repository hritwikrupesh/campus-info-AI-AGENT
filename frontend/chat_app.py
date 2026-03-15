import streamlit as st
import requests
import time

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(
    page_title="Smart Campus AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
def apply_custom_css():
    st.markdown("""
        <style>
        /* Base page background styling */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            font-family: 'Inter', sans-serif;
        }

        /* Remove default Streamlit top padding */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        /* Centered chat panel (styling the main Streamlit block container) */
        [data-testid="block-container"] {
            max-width: 950px;
            margin: auto;
            margin-top: 2rem;
            margin-bottom: 2rem;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.98);
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }

        /* Hide the top header added by Streamlit */
        header {
            visibility: hidden;
        }

        /* Title styling */
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.5rem;
        }

        /* Subtitle styling */
        .sub-title {
            text-align: center;
            font-size: 1.1rem;
            color: #64748b;
            margin-bottom: 2.5rem;
        }

        /* Message Bubbles */
        .user-message-container {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            margin-bottom: 1.5rem;
        }
        
        .ai-message-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            margin-bottom: 1.5rem;
        }

        /* Label above bubbles */
        .msg-label {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-bottom: 4px;
            font-weight: 600;
        }

        .user-bubble {
            background-color: #2563eb;
            color: white;
            padding: 1rem 1.25rem;
            border-radius: 18px 18px 0px 18px;
            max-width: 80%;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
            line-height: 1.5;
        }

        .ai-bubble {
            background-color: #f1f5f9;
            color: #1e293b;
            padding: 1rem 1.25rem;
            border-radius: 18px 18px 18px 0px;
            max-width: 80%;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            line-height: 1.5;
        }

        /* Input field styling adjustments */
        .stChatInputContainer {
            border-radius: 12px !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Helper function to render a single chat message
def render_message(role: str, content: str, sources: list = None):
    """Render a chat bubble using custom HTML."""
    if role == "user":
        html = f'''
        <div class="user-message-container">
            <div class="msg-label" style="text-align: right; width: 100%;">You</div>
            <div class="user-bubble">{content}</div>
        </div>
        '''
        st.markdown(html, unsafe_allow_html=True)
    else:
        html = build_ai_bubble_html(content)
        st.markdown(html, unsafe_allow_html=True)
        if sources:
            st.markdown("\n**Sources**")
            for url in sources:
                st.markdown(f"• [{url}]({url})")

def build_ai_bubble_html(content: str) -> str:
    """Build the HTML for the AI bubble (useful for iterative typing)."""
    return f'''
    <div class="ai-message-container">
        <div class="msg-label" style="text-align: left; width: 100%;">Campus AI</div>
        <div class="ai-bubble">{content}</div>
    </div>
    '''

def init_session_state():
    """Initialize chat history and trigger state in session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # State used to trigger a message from the sidebar examples
    if "sidebar_prompt" not in st.session_state:
        st.session_state.sidebar_prompt = None

def get_ai_response(query: str) -> dict:
    """Send query to backend API and return the full response data including sources."""
    try:
        response = requests.post(API_URL, json={"query": query}, timeout=400)
        response.raise_for_status()
        data = response.json()
        return {
            "answer": data.get("answer", "I received a response, but it didn't contain an answer."),
            "sources": data.get("sources", [])
        }
    except requests.exceptions.ConnectionError:
        return {"answer": "⚠️ Error: Could not connect to the Campus AI API. Please ensure the backend server is running.", "sources": []}
    except requests.exceptions.Timeout:
        return {"answer": "⚠️ Error: The request timed out. The server might be overloaded.", "sources": []}
    except Exception as e:
        return {"answer": f"⚠️ Error: An unexpected error occurred: {str(e)}", "sources": []}

def render_sidebar():
    """Render the sidebar with title, clear chat, examples, and branding."""
    with st.sidebar:
        st.markdown("## 🎓 Campus AI Assistant")
        st.markdown("Ask questions about courses, placements, faculty, and campus facilities.")
        st.divider()
        
        # Clear Chat Button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sidebar_prompt = None
            st.rerun()

        st.divider()
        st.markdown("### Example Questions")
        
        # Example questions buttons
        examples = [
            "Tell me about placements",
            "What courses are offered?",
            "What facilities are available on campus?",
            "Who are the faculty members?"
        ]
        
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state.sidebar_prompt = ex
                
        st.divider()
        
        # Branding Footer
        st.markdown(
            """
            <div style="text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 2rem;">
                <strong>Smart Campus AI</strong><br>
                Powered by RAG + TinyLlama
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    apply_custom_css()
    init_session_state()
    render_sidebar()
    
    # Header Section
    st.markdown('<div class="main-title">Smart Campus AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Ask questions about courses, placements, campus facilities, faculty, and more.</div>', unsafe_allow_html=True)

    # Use a container so the input always stays at the bottom natively
    chat_history_container = st.container()
    
    # Render all previous messages in history
    with chat_history_container:
        for msg in st.session_state.messages:
            render_message(msg["role"], msg["content"], msg.get("sources"))

    # Chat input box
    chat_input_prompt = st.chat_input("Ask a question about the campus...")
    
    # Check if a prompt came from the chat input or a sidebar click
    active_prompt = None
    if chat_input_prompt:
        active_prompt = chat_input_prompt
    elif st.session_state.sidebar_prompt:
        active_prompt = st.session_state.sidebar_prompt
        # Reset the trigger so it doesn't fire again on subsequent interactions
        st.session_state.sidebar_prompt = None 

    if active_prompt:
        # First, render the user message that was just submitted
        with chat_history_container:
            render_message("user", active_prompt)
            
            # Placeholder for the upcoming AI response
            ai_placeholder = st.empty()
            
            # Show a spinner while fetching data from API
            with st.spinner("Campus AI is thinking..."):
                response_data = get_ai_response(active_prompt)
                answer = response_data["answer"]
                sources = response_data.get("sources", [])
                
            # Typing animation: reveal the AI response progressively
            import re
            
            display_text = ""
            # Preserve all whitespace boundaries (like newlines) to keep formatting
            tokens = re.split(r'(\s+)', answer)
            
            if len(tokens) > 0:
                for token in tokens:
                    display_text += token
                    if token.strip(): # Only animate and sleep on actual words
                        ai_placeholder.markdown(build_ai_bubble_html(display_text), unsafe_allow_html=True)
                        time.sleep(0.03)
            else:
                # If the response is inherently empty (e.g. some whitespace error)
                ai_placeholder.markdown(build_ai_bubble_html(answer), unsafe_allow_html=True)
                
            if sources:
                st.markdown("\n**Sources**")
                for url in sources:
                    st.markdown(f"• [{url}]({url})")
            
            # Save the new interaction to session state
            st.session_state.messages.append({"role": "user", "content": active_prompt})
            st.session_state.messages.append({"role": "ai", "content": answer, "sources": sources})

if __name__ == "__main__":
    main()
