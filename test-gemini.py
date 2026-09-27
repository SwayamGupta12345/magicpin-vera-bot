from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

try:
    resp = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents='{"task": "test"}',
        config={
            "system_instruction": 'Reply with JSON: {"ok": true}',
            "response_mime_type": "application/json",
            "max_output_tokens": 100,
            "temperature": 0.3,
        },
    )
    print("SUCCESS")
    print(repr(resp.text))
except Exception as e:
    print("FAILED")
    print(type(e).__name__, str(e))