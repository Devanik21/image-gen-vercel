import os
from google import genai
from google.genai import types
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)

client = genai.Gemini(api_key=os.getenv("GOOGLE_API_KEY"))

@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["image"])
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
            return jsonify({"image": b64})
    return jsonify({"error": "no image returned"}), 500
