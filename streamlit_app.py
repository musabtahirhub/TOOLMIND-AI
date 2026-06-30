import streamlit as st
from google import genai
from google.genai import types
import requests
import math
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ToolMind AI Agent",
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
# TOOLS (Python functions)
# ─────────────────────────────────────────────

def web_search(query: str) -> str:
    """Search the web for current information on any topic using DuckDuckGo."""
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
    """Evaluate a mathematical expression. Use Python math syntax like '2 ** 10' or 'sqrt(144)'."""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["abs"] = abs
        allowed["round"] = round
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Could not calculate '{expression}': {str(e)}"


def get_weather(city: str) -> str:
    """Get the current weather for any city in the world."""
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


# Map tool names to their Python functions
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "calculate": calculate,
    "get_weather": get_weather,
}

TOOL_ICONS = {"web_search": "🔍", "calculate": "🧮", "get_weather": "🌤️"}


def run_tool(name: str, inputs: dict) -> str:
    """Execute a tool by name with the given inputs."""
    func = TOOL_FUNCTIONS.get(name)
    if func:
        return func(**inputs)
    return f"Unknown tool: {name}"


# ─────────────────────────────────────────────
# AGENT LOOP  (yields steps for live display)
# ─────────────────────────────────────────────

def run_agent_stream(user_message: str, api_key: str):
    """
    Generator that yields dicts describing each step of the agent loop.
    Types: "thinking", "tool_call", "tool_result", "answer", "error"
    Uses Google Gemini API with manual function calling.
    """
    client = genai.Client(api_key=api_key)

    # Define the tools for Gemini using the Python functions directly
    gemini_tools = [web_search, calculate, get_weather]

    # Build conversation contents
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
    ]

    max_iterations = 10  # Safety limit to prevent infinite loops

    for _ in range(max_iterations):
        yield {"type": "thinking", "text": "Gemini is thinking..."}

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=gemini_tools,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True,  # We handle tool calls manually for UI display
                    ),
                ),
            )
        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "401" in error_msg or "PERMISSION_DENIED" in error_msg:
                yield {"type": "error", "text": "❌ **Invalid API key.** Please check your Google Gemini API key and try again. Get one free at [aistudio.google.com](https://aistudio.google.com/apikey)"}
            else:
                yield {"type": "error", "text": f"❌ **API error:** {error_msg}"}
            return

        # Check if there are any function calls in the response
        has_function_calls = False
        function_call_parts = []
        text_parts = []

        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    has_function_calls = True
                    function_call_parts.append(part)
                elif part.text:
                    text_parts.append(part.text)

        if not has_function_calls:
            # No tool calls — extract the final text answer
            final_text = response.text if response.text else "".join(text_parts)
            if final_text:
                yield {"type": "answer", "text": final_text}
            else:
                yield {"type": "answer", "text": "Agent stopped unexpectedly."}
            return

        # Process function calls
        # Add the model's response (with function calls) to contents
        contents.append(response.candidates[0].content)

        # Execute each function call and collect results
        function_response_parts = []

        for part in function_call_parts:
            fc = part.function_call
            tool_name = fc.name
            tool_args = dict(fc.args) if fc.args else {}

            yield {"type": "tool_call", "name": tool_name, "input": tool_args}
            result = run_tool(tool_name, tool_args)
            yield {"type": "tool_result", "name": tool_name, "result": result}

            function_response_parts.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result},
                )
            )

        # Add tool results back to the conversation
        contents.append(
            types.Content(
                role="user",
                parts=function_response_parts,
            )
        )

    # If we hit the iteration limit
    yield {"type": "answer", "text": "Agent reached the maximum number of steps."}


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    # API key: prioritize st.secrets (Streamlit Cloud), then env var, then manual input
    default_key = ""
    try:
        default_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        default_key = os.getenv("GOOGLE_API_KEY", "")

    api_key = st.text_input(
        "Google Gemini API Key",
        value=default_key,
        type="password",
        help="Get yours free at aistudio.google.com/apikey",
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

    st.caption("Built with Gemini 2.5 Flash · [GitHub](https://github.com)")

# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────

st.title("🤖 ToolMind AI Agent")
st.caption("An autonomous agent that searches the web, does math, and checks weather — powered by Gemini 2.5 Flash (free).")

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
        st.error("Please enter your Google Gemini API key in the sidebar. Get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)")
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

            elif step["type"] == "error":
                tool_placeholder.empty()
                st.error(step["text"])
                st.stop()

            elif step["type"] == "answer":
                answer = step["text"]
                answer_placeholder.markdown(answer)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "steps": steps,
    })
