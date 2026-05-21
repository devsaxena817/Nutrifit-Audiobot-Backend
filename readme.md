# NutriFit AI Microservice

This project is a Flask-based AI microservice intended to be called by another backend such as Spring Boot.

## Endpoints

`GET /health`

Returns service status and the currently configured Gemini model.

`GET /models`

Lists Gemini models available to the current API key.

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

```powershell
.\venv2\Scripts\python.exe app.py
```
