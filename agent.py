import anthropic
import requests
import json
import math
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ─────────────────────────────────────────────
# TOOLS  (functions the agent can call)
# ─────────────────────────────────────────────

def web_search(query: str) -> str:
    """Search the web using DuckDuckGo (no API key needed)."""
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        results = []

        # Abstract (main answer)
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")

        # Related topics
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"- {topic['Text']}")

        if results:
            return "\n".join(results)
        else:
            return f"No direct answer found for '{query}'. Try rephrasing."
    except Exception as e:
        return f"Search failed: {str(e)}"


def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        # Only allow safe math operations
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["abs"] = abs
        allowed["round"] = round
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Could not calculate '{expression}': {str(e)}"


def get_weather(city: str) -> str:
    """Get current weather for a city using open-meteo (no API key needed)."""
    try:
        # Step 1: Geocode the city
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_resp = requests.get(geo_url, params={"name": city, "count": 1}, timeout=5)
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            return f"City '{city}' not found."

        loc = geo_data["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        name = loc["name"]

        # Step 2: Get weather
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weathercode,windspeed_10m",
            "timezone": "auto",
        }
        w_resp = requests.get(weather_url, params=params, timeout=5)
        w_data = w_resp.json()

        current = w_data["current"]
        temp = current["temperature_2m"]
        wind = current["windspeed_10m"]

        # Simple weather code mapping
        code = current["weathercode"]
        conditions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 51: "Light drizzle", 61: "Light rain", 71: "Light snow",
            80: "Rain showers", 95: "Thunderstorm",
        }
        condition = conditions.get(code, f"Weather code {code}")

        return f"Weather in {name}: {condition}, {temp}°C, wind {wind} km/h"
    except Exception as e:
        return f"Weather fetch failed: {str(e)}"


# ─────────────────────────────────────────────
# TOOL DEFINITIONS  (sent to Claude so it knows what tools exist)
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for current information on any topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use Python math syntax.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression e.g. '2 ** 10' or 'sqrt(144)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for any city in the world.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name e.g. 'London'"}
            },
            "required": ["city"],
        },
    },
]

# ─────────────────────────────────────────────
# TOOL ROUTER  (calls the right function)
# ─────────────────────────────────────────────

def run_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "web_search":
        return web_search(tool_input["query"])
    elif tool_name == "calculate":
        return calculate(tool_input["expression"])
    elif tool_name == "get_weather":
        return get_weather(tool_input["city"])
    else:
        return f"Unknown tool: {tool_name}"


# ─────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────

def run_agent(user_message: str) -> str:
    """
    Agentic loop:
    1. Send user message + tools to Claude
    2. If Claude wants to use a tool → run it → send result back
    3. Repeat until Claude gives a final text answer
    """
    print(f"\n{'='*50}")
    print(f"You: {user_message}")
    print(f"{'='*50}")

    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # If Claude is done → return the text answer
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            print(f"\nAgent: {final_text}")
            return final_text

        # If Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Add Claude's response to conversation history
            messages.append({"role": "assistant", "content": response.content})

            # Process each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n  [Tool call] {block.name}({block.input})")
                    result = run_tool(block.name, block.input)
                    print(f"  [Tool result] {result}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Send tool results back to Claude
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason
            break

    return "Agent stopped unexpectedly."


# ─────────────────────────────────────────────
# MAIN  (chat loop)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║        AI Agent  —  ready!           ║")
    print("║  Tools: web search, calculator,      ║")
    print("║         weather                      ║")
    print("║  Type 'quit' to exit                 ║")
    print("╚══════════════════════════════════════╝")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        run_agent(user_input)
