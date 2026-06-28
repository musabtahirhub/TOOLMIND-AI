import streamlit as st
import anthropic
import requests
import math
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="centered",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
.tool-badge {
    display: inline-block;
    background: #f0f2f6;
    border: 1px solid #d0d3da;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    font-family: monospace;
    color: #444;
    margin: 2px 2px;
}
.tool-result {
    background: #f8f9fb;
    border-left: 3px solid #4CAF50;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 13px;
    color: #333;
    margin: 4px 0;
}
.thinking-box {
    background: #fff8e1;
    border-left: 3px solid #FFC107;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 13px;
    color: #555;
    margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

def web_search(query: str) -> str:
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        results = []
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"- {topic['Text']}")
        return "\n".join(results) if results else f"No direct answer found for '{query}'."
    except Exception as e:
        return f"Search failed: {str(e)}"


def calculate(expression: str) -> str:
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["abs"] = abs
        allowed["round"] = round
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Could not calculate '{expression}': {str(e)}"


def get_weather(city: str) -> str:
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=5,
        )
        geo_data = geo_resp.json()
        if not geo_data.get("results"):
            return f"City '{city}' not found."
        loc = geo_data["results"][0]
        lat, lon, name = loc["latitude"], loc["longitude"], loc["name"]
        w_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,weathercode,windspeed_10m",
                "timezone": "auto",
            },
            timeout=5,
        )
        current = w_resp.json()["current"]
        conditions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 51: "Light drizzle", 61: "Light rain", 71: "Light snow",
            80: "Rain showers", 95: "Thunderstorm",
        }
        condition = conditions.get(current["weathercode"], f"Code {current['weathercode']}")
        return f"Weather in {name}: {condition}, {current['temperature_2m']}°C, wind {current['windspeed_10m']} km/h"
    except Exception as e:
        return f"Weather fetch failed: {str(e)}"


TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for current information on any topic.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use Python math syntax.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. '2 ** 10' or 'sqrt(144)'"}},
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for any city in the world.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name e.g. 'London'"}},
            "required": ["city"],
        },
    },
]

TOOL_ICONS = {"web_search": "🔍", "calculate": "🧮", "get_weather": "🌤️"}


def run_tool(name: str, inputs: dict) -> str:
    if name == "web_search":
        return web_search(inputs["query"])
    elif name == "calculate":
        return calculate(inputs["expression"])
    elif name == "get_weather":
        return get_weather(inputs["city"])
    return f"Unknown tool: {name}"


# ─────────────────────────────────────────────
# AGENT LOOP  (yields steps for live display)
# ─────────────────────────────────────────────

def run_agent_stream(user_message: str, api_key: str):
    """
    Generator that yields dicts describing each step of the agent loop.
    Types: "thinking", "tool_call", "tool_result", "answer"
    """
    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": user_message}]

    while True:
        yield {"type": "thinking", "text": "Claude is thinking..."}

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            final_text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            yield {"type": "answer", "text": final_text}
            return

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    yield {"type": "tool_call", "name": block.name, "input": block.input}
                    result = run_tool(block.name, block.input)
                    yield {"type": "tool_result", "name": block.name, "result": result}
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})
        else:
            yield {"type": "answer", "text": "Agent stopped unexpectedly."}
            return


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    # API key input
    api_key = st.text_input(
        "Anthropic API Key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Get yours at console.anthropic.com",
    )

    st.divider()

    st.markdown("**Available tools**")
    st.markdown("🔍 **Web search** — DuckDuckGo")
    st.markdown("🧮 **Calculator** — safe math eval")
    st.markdown("🌤️ **Weather** — Open-Meteo API")

    st.divider()

    st.markdown("**Try these:**")
    examples = [
        "What is the capital of Japan?",
        "What's the weather in Lahore?",
        "What is 2 to the power of 32?",
        "Search for Python and tell me about it",
        "Pakistan population × 2%",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["prefill"] = ex

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Built with Claude claude-sonnet-4-6 · [GitHub](https://github.com)")

# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────

st.title("🤖 AI Agent")
st.caption("An autonomous agent that searches the web, does math, and checks weather.")

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Show tool steps first, then answer
            for step in msg.get("steps", []):
                if step["type"] == "tool_call":
                    icon = TOOL_ICONS.get(step["name"], "🔧")
                    params = ", ".join(f"{k}={v!r}" for k, v in step["input"].items())
                    st.markdown(
                        f'<span class="tool-badge">{icon} {step["name"]}({params})</span>',
                        unsafe_allow_html=True,
                    )
                elif step["type"] == "tool_result":
                    st.markdown(
                        f'<div class="tool-result">{step["result"]}</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown(msg["content"])
        else:
            st.markdown(msg["content"])

# Handle prefilled input from sidebar buttons
prefill = st.session_state.pop("prefill", "")

# Chat input
user_input = st.chat_input("Ask me anything...") or prefill

if user_input:
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent and display steps live
    with st.chat_message("assistant"):
        steps = []
        answer = ""

        tool_placeholder = st.empty()
        answer_placeholder = st.empty()

        tool_html = ""

        for step in run_agent_stream(user_input, api_key):
            if step["type"] == "thinking":
                tool_html += f'<div class="thinking-box">💭 {step["text"]}</div>'
                tool_placeholder.markdown(tool_html, unsafe_allow_html=True)

            elif step["type"] == "tool_call":
                icon = TOOL_ICONS.get(step["name"], "🔧")
                params = ", ".join(f"{k}={v!r}" for k, v in step["input"].items())
                tool_html += f'<span class="tool-badge">{icon} {step["name"]}({params})</span><br>'
                tool_placeholder.markdown(tool_html, unsafe_allow_html=True)
                steps.append(step)

            elif step["type"] == "tool_result":
                tool_html += f'<div class="tool-result">↳ {step["result"]}</div>'
                tool_placeholder.markdown(tool_html, unsafe_allow_html=True)
                steps.append(step)

            elif step["type"] == "answer":
                answer = step["text"]
                answer_placeholder.markdown(answer)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "steps": steps,
    })
