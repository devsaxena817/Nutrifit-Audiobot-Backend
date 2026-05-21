# NutriFit AI Microservice

This project is a Flask-based AI microservice intended to be called by another backend such as Spring Boot.

## Endpoints

`GET /health`

Returns service status and the currently configured Gemini model.

`GET /models`

Returns the configured Gemini model.

`POST /analyze`

Accepts multipart form-data:

- `audio`: required audio file
- `include_pdf`: optional, set to `true` to include a base64-encoded PDF in the JSON response

Response fields:

- `json`: structured nutrition analysis
- `report_text`: readable report text
- `model_name`: selected Gemini model
- `pdf_base64`: optional base64 PDF
- `pdf_filename`: optional PDF filename

## Run

Set the required environment variables first:

```env
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=models/gemini-2.5-flash
SERVICE_API_TOKEN=your_secret_token_shared_with_your_backend
MAX_UPLOAD_MB=25
```

`GEMINI_MODEL` is optional. If it is not set, the service uses `models/gemini-2.5-flash`.

```powershell
.\venv2\Scripts\python.exe app.py
```
