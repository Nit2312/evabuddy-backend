"""Flask entrypoint for Render (and local dev).

Exports the Flask `app` from api/app.py.
"""

from api.app import app

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
