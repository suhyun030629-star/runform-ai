"""WSGI entry point for RunForm AI.

Docker/Gunicorn and local development both use the same Flask application.
"""

from flask_app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8510, debug=False)
