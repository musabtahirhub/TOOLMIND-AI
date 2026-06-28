# AI Agent with Tool Use

An autonomous AI agent built with the **Anthropic Claude API** that can search the web, perform calculations, and fetch real-time weather — all by deciding which tool to use on its own.

This project demonstrates a core pattern in modern AI engineering: the **agentic loop**, where an LLM dynamically selects and chains tools to answer complex questions.

---

## Demo

```
You: Search for Pakistan's population and calculate what 2% of it is

  [Tool call] web_search({'query': 'Pakistan population 2024'})
  [Tool result] Summary: Pakistan's population is approximately 240 million...

  [Tool call] calculate({'expression': '240000000 * 0.02'})
  [Tool result] 240000000 * 0.02 = 4800000.0

Agent: Pakistan's population is approximately 240 million.
       2% of that equals 4,800,000 people.
```

---

## Features

- **Web search** — queries DuckDuckGo (no API key required)
- **Calculator** — safely evaluates math expressions using Python's `math` module
- **Real-time weather** — fetches live data via Open-Meteo (no API key required)
- **Agentic loop** — Claude autonomously chains multiple tools when needed
- **Conversation history** — maintains full context across multi-turn interactions

---

## How it works

```
You → Claude reads message + available tools
         ↓
   Does Claude need a tool?
   ├── YES → calls tool → gets result → sends back to Claude → repeat
   └── NO  → writes final answer → done
```

Claude never runs code itself. It outputs a structured request ("call `get_weather` with `city=Lahore`"), your Python code executes it, and the result is fed back. This loop repeats until Claude has enough information to answer.

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Claude claude-sonnet-4-6 (Anthropic) |
| Web search | DuckDuckGo Instant Answer API |
| Weather | Open-Meteo + Geocoding API |
| Calculator | Python `math` + safe `eval` |
| Secret management | `python-dotenv` |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ai-agent.git
cd ai-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get an Anthropic API key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up and create an API key
3. New accounts get free credits to start

### 5. Create your `.env` file

```bash
cp .env.example .env
```

Then open `.env` and paste your real API key:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 6. Run the agent

```bash
python agent.py
```

---

## Example queries

| Query | Tools used |
|---|---|
| `What is the capital of Japan?` | web_search |
| `What is 2 to the power of 32?` | calculate |
| `What's the weather in Lahore?` | get_weather |
| `What is 15% of 8500?` | calculate |
| `Search Tesla stock news and summarize` | web_search |
| `Pakistan population × 2%` | web_search → calculate |

---

## Project structure

```
ai-agent/
├── streamlit_app.py      # Web UI (Streamlit)
├── agent.py              # CLI version with agentic loop
├── requirements.txt      # Python dependencies
├── .env.example          # Template for environment variables
├── .env                  # Your secrets (never committed to Git)
├── .gitignore            # Prevents secrets from being uploaded
├── .streamlit/
│   └── config.toml       # Streamlit theme and server config
└── README.md             # This file
```

---

## Key concepts demonstrated

- **Tool use / function calling** — defining tools as JSON schemas and letting the LLM select them
- **Agentic loop** — `while True` loop that runs until `stop_reason == "end_turn"`
- **Conversation history** — passing the full message list to maintain context
- **Safe eval** — sandboxed math expression evaluation
- **Error handling** — `try/except` on all external API calls
- **Environment variables** — secret management with `python-dotenv`

---

## Deployment

### Streamlit Community Cloud (recommended — free, live URL)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select this repo → set **Main file** to `streamlit_app.py`
4. Click **Advanced settings** → add your secret:
   ```
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
5. Click **Deploy** — you'll get a live URL in ~60 seconds

### Run locally with Streamlit UI

```bash
streamlit run streamlit_app.py
```

### Run as CLI (original terminal version)

```bash
python agent.py
```

---

## Future improvements

- [ ] Add Wikipedia tool
- [ ] Add news search tool
- [ ] Add memory/persistence between sessions
- [ ] Build a Streamlit web UI
- [ ] Add streaming responses

---

## Author

Built as a portfolio project demonstrating AI agent architecture with Claude claude-sonnet-4-6.
