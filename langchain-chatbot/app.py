import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import requests

# Configure Streamlit page
st.set_page_config(
    page_title="LangChain Ollama Chatbot",
    page_icon="🤖",
    layout="centered"
)
#Charan
# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to check if Ollama is running
def check_ollama_connection():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Function to initialize the LLM and chain
@st.cache_resource
def initialize_chain(model_name="llama3"):
    # Create the ChatOllama instance
    llm = ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.7,
    )

    # Create a prompt template with system message and conversation history
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a helpful, friendly AI assistant.
You provide clear, accurate, and thoughtful responses to user questions.
You remember the conversation history and can refer back to previous messages.
Be conversational and engaging while remaining professional."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create the LCEL chain
    chain = prompt | llm | StrOutputParser()

    return chain

# Function to format chat history for the chain
def get_chat_history():
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history

# Main app UI
st.title("🤖 LangChain Chatbot with Ollama")
st.caption("Powered by LangChain, Ollama, and Streamlit")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Check Ollama connection status
    if check_ollama_connection():
        st.success("✅ Ollama is running")
    else:
        st.error("❌ Ollama is not running")
        st.info("""
        To start Ollama:
        1. Install: `curl -fsSL https://ollama.com/install.sh | sh`
        2. Run: `ollama serve`
        3. Pull a model: `ollama pull llama3`
        """)
        st.stop()

    # Model selection
    model_name = st.selectbox(
        "Select Model",
        ["llama3", "mistral", "llama3.2", "gemma2"],
        index=0,
        help="Choose the Ollama model to use"
    )

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Display conversation stats
    st.metric("Messages", len(st.session_state.messages))

    st.divider()

    st.markdown("""
    ### About
    This chatbot uses:
    - **LangChain** for orchestration
    - **Ollama** for local LLMs
    - **Streamlit** for the UI

    The bot remembers your conversation history within the session.
    """)

# Initialize the chain
try:
    chain = initialize_chain(model_name)
except Exception as e:
    st.error(f"Error initializing the model: {str(e)}")
    st.info("Make sure you've pulled the model with `ollama pull llama3`")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            # Get chat history and invoke the chain
            chat_history = get_chat_history()[:-1]  # Exclude the current message

            with st.spinner("Thinking..."):
                response = chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history
                })

            # Display the response
            message_placeholder.markdown(response)

            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
st.caption("Built with LangChain 0.3+ and Ollama | Conversation memory enabled")
