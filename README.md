# evabuddy-backend

## Deploying on Render

Use **Python 3.11** (not 3.14). LangChain’s Pydantic V1 compatibility and some transitive deps (e.g. around `exception_key` type inference) can fail on 3.14.

- In the Render service **Environment** tab, set: `PYTHON_VERSION=3.11.11`
- Or ensure a `.python-version` file in the build root contains `3.11.11`

Build command: `pip install -r requirements.txt`  
Start command: `gunicorn app:app` (run from the backend root so `app` is `app.py`).
