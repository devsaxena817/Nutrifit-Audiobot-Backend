import os
import re
import json
import tempfile
import traceback
import base64
from io import BytesIO
from functools import wraps

from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load env
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY in your .env")

SERVICE_API_TOKEN = os.getenv("SERVICE_API_TOKEN")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))

genai.configure(api_key=API_KEY)

DEFAULT_GEMINI_MODEL = "models/gemini-2.5-flash"
MODEL_NAME = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


model = genai.GenerativeModel(model_name=MODEL_NAME)
print(f"Selected Gemini model: {MODEL_NAME}")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_AUDIO_MIME_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/x-m4a",
    "audio/webm",
    "audio/ogg",
}


DUAL_PROMPT = r"""
You are NutriFit AI — an advanced medical & nutrition intelligence assistant that analyzes dietician-client voice consultations.

TASK:
1) Transcribe the audio.
2) Extract structured health & diet insights.
3) Return two parts in this exact order:

PART A — JSON ONLY (MANDATORY, parseable):
Return a single JSON object with these keys exactly:
{
  "transcript": "full transcription string",
  "summary": "short 4-6 line summary",
  "key_health_concerns": [{"label":"string","evidence":"text excerpt","confidence":0.0}],
  "dietary_habits": [{"label":"string","details":"text","confidence":0.0}],
  "allergies_or_restrictions": [{"label":"string","evidence":"text","confidence":0.0}],
  "suggested_improvements": ["action item 1","action item 2"],
  "personalized_nutrition": {
      "calorie_target": "e.g. 1800 kcal/day or null",
      "macro_split": {"protein_pct":30,"carb_pct":45,"fat_pct":25},
      "sample_meal_plan": ["Breakfast: ...","Lunch: ..."],
      "hydration_l_per_day": 2.5,
      "supplements": ["name - reason"]
  },
  "tone_emotion": {"primary":"Stressed","secondary":["Anxious"], "confidence":0.0},
  "follow_up_questions": ["question 1","question 2"],
  "metadata": {"duration_seconds": null, "speaker_segments": [], "confidence_overall":0.0}
}

Important rules:
- JSON must be valid JSON only (no backticks, no explanation before it). If any field is unknown use null / [] / "".
- Confidence fields are floats 0.0–1.0.

PART B — HUMAN-READABLE REPORT (after the JSON):
Provide a concise professional report using the same headings. Keep it scannable.

END.
"""


def require_service_token(view_func):
    """
    Optional server-to-server protection.
    If SERVICE_API_TOKEN is set, callers must send: Authorization: Bearer <token>
    """
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not SERVICE_API_TOKEN:
            return view_func(*args, **kwargs)

        auth_header = request.headers.get("Authorization", "")
        expected = f"Bearer {SERVICE_API_TOKEN}"
        if auth_header != expected:
            return jsonify({"error": "Unauthorized"}), 401

        return view_func(*args, **kwargs)

    return wrapper


def call_model_with_audio(audio_bytes: bytes, mime_type: str, prompt: str) -> str:
    """
    Upload audio via the Files API, then call Gemini generate_content.
    """
    temp_path = None
    uploaded_file = None

    # The Files API is the most reliable path for audio inputs across Gemini models.
    suffix = mimetype_to_suffix(mime_type)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        uploaded_file = genai.upload_file(temp_path, mime_type=mime_type)
        response = model.generate_content([prompt, uploaded_file])
        return response.text
    finally:
        if uploaded_file is not None:
            try:
                genai.delete_file(uploaded_file.name)
            except Exception:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def mimetype_to_suffix(mime_type: str) -> str:
    """
    Preserve a reasonable file extension for uploaded temp audio files.
    """
    mapping = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
    }
    return mapping.get(mime_type, ".bin")


def extract_first_json(text: str):
    """
    Extract the first JSON object from model output robustly using recursion-capable regex.
    Returns Python object or None.
    """
    # Try a regex using balanced-braces via recursion if supported
    # Fallback simpler approach if regex engine doesn't support recursion
    # We'll search for the first '{' and then attempt to parse progressively until valid JSON parsed.
    start = text.find("{")
    if start == -1:
        return None, text

    # Try to find a matching closing bracket by expanding
    for end in range(start + 1, len(text)):
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            # success
            remainder = text[end + 1:].strip()
            return parsed, remainder
        except Exception:
            continue

    # As fallback, try to find a block using a looser regex (may fail on nested)
    m = re.search(r'(\{(?:[^{}]|\{[^{}]*\})*\})', text, flags=re.DOTALL)
    if m:
        raw = m.group(1)
        try:
            return json.loads(raw), (text.replace(raw, "", 1).strip())
        except Exception:
            return None, text

    return None, text


def validate_json_schema(j: dict) -> bool:
    """
    Minimal validation: ensure essential keys exist.
    """
    required = ["transcript", "summary", "personalized_nutrition"]
    return all(k in j for k in required)


def create_pdf_bytes_from_json(j: dict) -> bytes:
    """
    Create a readable PDF from structured JSON and return it as bytes.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    margin_x = 40
    y = height - 50
    line_h = 14

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_x, y, "NutriFit AI - Consultation Report")
    y -= 28

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, y, "Summary:")
    y -= line_h
    c.setFont("Helvetica", 11)
    for ln in (j.get("summary") or "").splitlines():
        if y < 60:
            c.showPage()
            y = height - 50
        c.drawString(margin_x, y, ln)
        y -= line_h

    # Helper to draw a section
    def draw_section(title, content_lines):
        nonlocal y
        if y < 80:
            c.showPage()
            y = height - 50
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin_x, y, title)
        y -= line_h
        c.setFont("Helvetica", 11)
        for line in content_lines:
            if y < 60:
                c.showPage()
                y = height - 50
            c.drawString(margin_x + 6, y, "- " + line)
            y -= line_h

    # Key health concerns
    kh = []
    for item in j.get("key_health_concerns", []):
        lab = item.get("label", "")
        ev = item.get("evidence", "")
        kh.append(f"{lab} — {ev} (conf: {item.get('confidence',0):.2f})")
    if kh:
        draw_section("Key Health Concerns", kh)

    # Dietary habits
    dh = []
    for item in j.get("dietary_habits", []):
        dh.append(f"{item.get('label','')}: {item.get('details','')} (conf: {item.get('confidence',0):.2f})")
    if dh:
        draw_section("Dietary Habits", dh)

    # Suggestions
    sug = j.get("suggested_improvements", [])
    if sug:
        draw_section("Suggested Improvements", sug)

    # Personalized nutrition
    pn = j.get("personalized_nutrition", {})
    p_lines = []
    p_lines.append(f"Calorie target: {pn.get('calorie_target') or 'N/A'}")
    ms = pn.get("macro_split") or {}
    p_lines.append(f"Macro split: P {ms.get('protein_pct','-')}% | C {ms.get('carb_pct','-')}% | F {ms.get('fat_pct','-')}%")
    if pn.get("hydration_l_per_day"):
        p_lines.append(f"Hydration: {pn.get('hydration_l_per_day')} L/day")
    for meal in pn.get("sample_meal_plan", []):
        p_lines.append(meal)
    if pn.get("supplements"):
        p_lines.append("Supplements: " + ", ".join(pn.get("supplements")))
    draw_section("Personalized Nutrition", p_lines)

    c.save()
    buffer.seek(0)
    return buffer.read()


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "nutrifit-ai",
            "message": "NutriFit AI microservice is running. Use /health, /models, or POST /analyze.",
            "endpoints": {
                "health": "/health",
                "models": "/models",
                "analyze": "/analyze",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "nutrifit-ai",
            "model_name": MODEL_NAME,
        }
    )


@app.route("/models", methods=["GET"])
def models_endpoint():
    return jsonify(
        {
            "model_name": MODEL_NAME,
        }
    )


@app.route("/analyze", methods=["POST"])
@require_service_token
def analyze():
    """
    Receives an audio file (form-data key: audio), calls Gemini, and returns
    structured JSON plus a human-readable report. Optional form field:
    include_pdf=true to embed a base64 PDF in the response.
    """
    f = request.files.get("audio")
    if not f:
        return jsonify({"error": "No audio file received"}), 400

    audio_bytes = f.read()
    mime_type = f.content_type or "audio/wav"
    if mime_type not in ALLOWED_AUDIO_MIME_TYPES:
        return jsonify({"error": "Unsupported audio type", "mime_type": mime_type}), 415

    if not audio_bytes:
        return jsonify({"error": "Uploaded audio file is empty"}), 400

    # Call model
    try:
        raw = call_model_with_audio(audio_bytes, mime_type, DUAL_PROMPT)
    except Exception as e:
        print(
            "Model call failed:",
            {
                "filename": f.filename,
                "mime_type": mime_type,
                "size_bytes": len(audio_bytes),
                "model_name": MODEL_NAME,
                "error": str(e),
            },
        )
        traceback.print_exc()
        return jsonify(
            {
                "error": "Model call failed",
                "details": str(e),
                "model_name": MODEL_NAME,
            }
        ), 500

    # Extract JSON
    parsed_json, remainder = extract_first_json(raw)
    if parsed_json is None:
        # Return raw for debugging
        return jsonify({"error": "Could not extract JSON from model output", "raw": raw}), 500

    # Minimal validation
    if not validate_json_schema(parsed_json):
        # still continue but flagged
        parsed_json["_validation_warning"] = "Missing required top-level keys"

    response_payload = {
        "json": parsed_json,
        "report_text": remainder.strip() or raw,
        "model_name": MODEL_NAME,
    }

    include_pdf = (request.form.get("include_pdf") or "").strip().lower() == "true"
    if include_pdf:
        try:
            pdf_bytes = create_pdf_bytes_from_json(parsed_json)
            response_payload["pdf_base64"] = base64.b64encode(pdf_bytes).decode("ascii")
            response_payload["pdf_filename"] = "NutriFit_Report.pdf"
        except Exception as exc:
            response_payload["pdf_error"] = str(exc)

    return jsonify(response_payload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
