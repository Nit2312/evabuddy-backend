"""Flask entrypoint for Render (and local dev).

Exports the Flask `app` from api/app.py.
"""

from api.app import app

if __name__ == "__main__":
    # Flask dev server: make it explicitly threaded so concurrent requests
    # don't block each other during I/O-bound work.
    port = int(__import__("os").getenv("PORT", "5001"))
    app.run(debug=True, host="0.0.0.0", port=port, threaded=True)
