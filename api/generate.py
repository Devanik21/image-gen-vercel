import os, json, base64
from google import genai
from google.genai import types

# Init the Gemini client
client = genai.Gemini(api_key=os.getenv("GOOGLE_API_KEY"))

def handler(request):
    # Parse the incoming JSON
    data = json.loads(request.body)
    prompt = data.get("prompt", "")

    # Generate an image
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["image"]
        )
    )

    # Extract the first inline image and base64‑encode it
    b64 = None
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
            break

    if not b64:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "no image returned"})
        }

    # Return JSON with CORS headers so your front‑end can read it
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({"image": b64})
    }
