from google import genai
from google.genai import types
import requests
import json
import math
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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
    """Safely evaluate a math expression. Use Python math syntax like '2 ** 10' or 'sqrt(144)'."""
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


# Map tool names to functions
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "calculate": calculate,
    "get_weather": get_weather,
}


def run_tool(tool_name: str, tool_input: dict) -> str:
    func = TOOL_FUNCTIONS.get(tool_name)
    if func:
        return func(**tool_input)
    return f"Unknown tool: {tool_name}"


# ─────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────

def run_agent(user_message: str) -> str:
    """
    Agentic loop:
    1. Send user message + tools to Gemini
    2. If Gemini wants to use a tool → run it → send result back
    3. Repeat until Gemini gives a final text answer
    """
    print(f"\n{'='*50}")
    print(f"You: {user_message}")
    print(f"{'='*50}")

    # Define tools as the Python functions themselves
    gemini_tools = [web_search, calculate, get_weather]

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
    ]

    max_iterations = 10

    for _ in range(max_iterations):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
            ),
        )

        # Check for function calls
        has_function_calls = False
        function_call_parts = []

        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    has_function_calls = True
                    function_call_parts.append(part)

        # If no tool calls → return the text answer
        if not has_function_calls:
            final_text = response.text or "Agent stopped unexpectedly."
            print(f"\nAgent: {final_text}")
            return final_text

        # Process function calls
        contents.append(response.candidates[0].content)

        function_response_parts = []
        for part in function_call_parts:
            fc = part.function_call
            tool_name = fc.name
            tool_args = dict(fc.args) if fc.args else {}

            print(f"\n  [Tool call] {tool_name}({tool_args})")
            result = run_tool(tool_name, tool_args)
            print(f"  [Tool result] {result}")

            function_response_parts.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result},
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=function_response_parts,
            )
        )

    return "Agent reached the maximum number of steps."


# ─────────────────────────────────────────────
# MAIN  (chat loop)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║     ToolMind AI Agent  —  ready!     ║")
    print("║  Model: Gemini 2.5 Flash (free)      ║")
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
