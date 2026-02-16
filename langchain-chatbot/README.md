# LangChain Ollama Chatbot

A conversational AI chatbot built with LangChain, Ollama, and Streamlit. Features persistent conversation memory within sessions and runs completely locally using Ollama's LLMs.

## Features

- 🤖 **Local LLM Integration**: Uses Ollama for completely local AI inference
- 💬 **Conversation Memory**: Remembers context throughout the chat session
- 🎨 **Clean UI**: Streamlit-based chat interface
- ⚡ **LCEL Chains**: Built with modern LangChain Expression Language
- 🔄 **Model Selection**: Switch between different Ollama models
- 🗑️ **Clear Chat**: Reset conversation anytime
- ⚠️ **Error Handling**: Graceful handling when Ollama is unavailable

## Tech Stack

- **Python 3.11+**
- **LangChain 0.3+** - Orchestration and memory management
- **Ollama** - Local LLM backend (Llama 3, Mistral, etc.)
- **Streamlit** - Web UI framework

## Prerequisites

- Python 3.11 or higher
- macOS (for this setup)

## Installation

### 1. Install Ollama

Download and install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Or download directly from [ollama.com](https://ollama.com)

### 2. Start Ollama Service

```bash
ollama serve
```

Leave this running in a terminal. Ollama will run on `http://localhost:11434`

### 3. Pull an LLM Model

In a new terminal, pull at least one model:

```bash
# Recommended: Llama 3 (4.7GB)
ollama pull llama3

# Or Mistral (4.1GB)
ollama pull mistral

# Or smaller Llama 3.2 (2.0GB)
ollama pull llama3.2
```

### 4. Set Up Python Environment

Clone or navigate to this project directory:

```bash
cd langchain-chatbot
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Chatbot

With your virtual environment activated and Ollama running:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Check Connection**: The sidebar shows Ollama connection status
2. **Select Model**: Choose from available models in the dropdown
3. **Chat**: Type messages in the input box at the bottom
4. **Clear History**: Click "Clear Chat" to start a new conversation
5. **View Stats**: See message count in the sidebar

### Features in Action

**Conversation Memory Example:**
```
You: My name is Alice
Bot: Nice to meet you, Alice! How can I help you today?

You: What's my name?
Bot: Your name is Alice, as you just told me!
```

The bot remembers previous messages in the conversation.

## Project Structure

```
langchain-chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## How It Works

### Architecture

1. **LangChain LCEL Chain**: Uses LangChain Expression Language for building the conversational pipeline
   - `ChatPromptTemplate`: Defines system prompt and message structure
   - `ChatOllama`: Interfaces with Ollama's API
   - `StrOutputParser`: Parses LLM output to string

2. **Conversation Memory**:
   - Stores messages in Streamlit session state
   - Converts to LangChain message format (HumanMessage, AIMessage)
   - Passes full history to each LLM invocation

3. **Streamlit UI**:
   - Chat interface with message history
   - Real-time status checking
   - Model configuration

### Key Components

**Chain Definition** (`app.py`):
```python
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="..."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()
```

**Memory Management**:
- Messages stored as `{"role": "user/assistant", "content": "..."}`
- Converted to LangChain message objects for each inference
- Persists within Streamlit session (cleared on refresh or button click)

## Troubleshooting

### "Ollama is not running"

Make sure Ollama is started:
```bash
ollama serve
```

### "Model not found"

Pull the model first:
```bash
ollama pull llama3
```

### "Connection refused"

Verify Ollama is running on port 11434:
```bash
curl http://localhost:11434/api/tags
```

### Slow responses

- First response is slower (model loading)
- Larger models (70B+) require more RAM and are slower
- Try smaller models like `llama3.2` or `mistral`

## Available Models

Popular Ollama models you can use:

- `llama3` - Meta's Llama 3 (4.7GB) - Recommended
- `mistral` - Mistral 7B (4.1GB)
- `llama3.2` - Smaller Llama 3.2 (2.0GB)
- `gemma2` - Google's Gemma 2 (5.4GB)
- `phi3` - Microsoft's Phi-3 (2.3GB)

List all pulled models:
```bash
ollama list
```

## Customization

### Change the System Prompt

Edit the `SystemMessage` content in `app.py` to change the bot's personality:

```python
SystemMessage(content="Your custom personality here...")
```

### Add More Models

Edit the model selector in the sidebar:

```python
model_name = st.selectbox(
    "Select Model",
    ["llama3", "mistral", "your-model"],  # Add here
    index=0
)
```

### Adjust Temperature

Change the creativity/randomness (0.0 = deterministic, 1.0 = creative):

```python
llm = ChatOllama(
    model=model_name,
    temperature=0.7,  # Adjust this
)
```

## License

MIT

## Contributing

Feel free to open issues or submit pull requests!

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.com)
- [Streamlit Documentation](https://docs.streamlit.io/)
