# Industrial Intelligence Copilot — Local Run Guide

This guide explains how to run the project after cloning the repository.

## Prerequisites

Install these before starting:

- Git — needed to clone the repository
- Python 3.11.x
- Node.js 18.16 or newer
- npm — included automatically with Node.js
- Internet connection — required during the first installation and for optional Groq AI responses

Check that they are installed:

```powershell
python --version
node --version
npm --version
git --version
```

Recommended tested versions:

```text
Python 3.11.4
Node.js 18.16.1
```

No Docker, Poetry, uv, database, Jupyter, or globally installed Python/Node packages are required.

## 1. Clone the repository

```powershell
git clone ...(http/ssh)
cd ...(folder directory in which you cloned it)
```

Replace the URL with the actual GitHub repository URL.

## 2. Check required project files

Run:

```powershell
Test-Path data\raw\ai4i2020.csv
Test-Path models\training_metrics.json
```

Both commands should return:

```text
True
```

If the data file is missing, run:

```powershell
python scripts\download_data.py
```

If the model files are missing, complete Python setup first, then run:

```powershell
.\.venv\Scripts\python.exe scripts\train_models.py
```

## 3. Create the Python environment

```powershell
py -3.11 -m venv .venv
```

If `py -3.11` does not work, use:

```powershell
python -m venv .venv
```

## 4. Install Python dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

## 5. Configure `.env`

### Case A — `.env` already exists and contains a valid Groq key

No extra configuration is needed. Continue to Step 6.

### Case B — `.env` does not exist

Create it:

```powershell
Copy-Item .env.example .env
```

Then open it:

```powershell
notepad .env
```

### Case C — use Groq AI

Set these values in `.env`:

```dotenv
LLM_ENABLED=true
LLM_PROVIDER=groq
GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_MODEL=openai/gpt-oss-120b
GROQ_TIMEOUT_SECONDS=20
```

Test Groq:

```powershell
.\.venv\Scripts\python.exe scripts\check_groq.py
```

Expected result:

```text
Result: AVAILABLE
```

### Case D — no Groq key, invalid key, or Groq free-tier rate limit

Set:

```dotenv
LLM_ENABLED=false
LLM_PROVIDER=groq
GROQ_API_KEY=
```

The project still works normally.

The dashboard, data analysis, ML risk model, scenarios, charts, incidents, what-if analysis, and maintenance workflow remain available.

The Copilot will return verified deterministic evidence instead of a Groq-generated response.

## 6. Install frontend dependencies

```powershell
cd frontend
npm ci
cd ..
```

If `npm ci` fails, use:

```powershell
cd frontend
npm install
cd ..
```

## 7. Optional verification

Run backend tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected:

```text
87 passed
```

Run code checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

Expected:

```text
All checks passed!
```

Build the frontend:

```powershell
cd frontend
npm run build
cd ..
```

## 8. Start the application

Open two PowerShell terminals.

### Terminal 1 — backend

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn industrial_copilot.api.main:app --reload
```

Expected:

```text
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

Backend URLs:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

### Terminal 2 — frontend

From the repository root:

```powershell
cd frontend
npm run dev
```

Expected:

```text
Local: http://localhost:5173/
```

Open the application:

```text
http://localhost:5173/
```


## Common issues

### `python` or `py` is not recognized

Install Python 3.11 and select **Add Python to PATH** during installation.

### `node` or `npm` is not recognized

Install Node.js LTS, then close and reopen PowerShell or VS Code.

### Port 8000 is already in use

Stop the older backend terminal with `Ctrl + C`, then start the backend again.

### Port 5173 is already in use

Stop the older frontend terminal with `Ctrl + C`, then run:

```powershell
cd frontend
npm run dev
```

### Groq answer shows `Verified fallback`

This is expected when Groq is disabled, the key is unavailable, the key is invalid, or the Groq free-tier limit is temporarily reached. The application remains usable.
