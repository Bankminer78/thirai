import os
import secrets
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv

# LiveKit server-side token utilities
try:
    # livekit Python server SDK
    from livekit import AccessToken, VideoGrant  # type: ignore
except Exception as exc:  # pragma: no cover
    AccessToken = None  # type: ignore
    VideoGrant = None  # type: ignore


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
ROOM_NAME = os.getenv("ROOM_NAME", "thirai-conference")
PORT = int(os.getenv("PORT", "5057"))

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path="")


@app.get("/")
def index():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.get("/get_token")
def get_token():
    """Mint a LiveKit access token for a single shared room.

    Query params:
      - identity (optional): unique user identity; if missing, a random one is generated
      - name (optional): display name (defaults to identity)
    """
    if not (LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        return (
            jsonify({
                "error": "LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET must be set",
            }),
            500,
        )

    if AccessToken is None or VideoGrant is None:
        return jsonify({"error": "livekit Python package not installed"}), 500

    identity: str = request.args.get("identity") or f"user-{secrets.token_hex(4)}"
    display_name: str = request.args.get("name") or identity

    # Create a token that allows joining and publishing/subscribing to the single room
    token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, identity=identity, name=display_name)
    grant = VideoGrant(
        room=ROOM_NAME,
        room_join=True,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )
    token.add_grant(grant)
    jwt = token.to_jwt()

    return jsonify({
        "token": jwt,
        "url": LIVEKIT_URL,
        "room": ROOM_NAME,
        "identity": identity,
        "name": display_name,
    })


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})


# Lightweight diagnostics to help troubleshoot issues without exposing secrets
@app.get("/diag")
def diag():
    diagnostics = {
        "env": {
            "livekit_url_set": bool(LIVEKIT_URL),
            "api_key_set": bool(LIVEKIT_API_KEY),
            "api_secret_set": bool(LIVEKIT_API_SECRET),
            "room": ROOM_NAME,
        },
        "static": {
            "index_exists": os.path.exists(os.path.join(PUBLIC_DIR, "index.html")),
            "app_js_exists": os.path.exists(os.path.join(PUBLIC_DIR, "js", "app.js")),
            "style_exists": os.path.exists(os.path.join(PUBLIC_DIR, "style.css")),
        },
        "token": {
            "ok": False,
            "error": None,
        },
    }

    if AccessToken is None or VideoGrant is None:
        diagnostics["token"]["ok"] = False
        diagnostics["token"]["error"] = "livekit Python package not installed"
        return jsonify(diagnostics), 200

    if not (LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        diagnostics["token"]["ok"] = False
        diagnostics["token"]["error"] = "Missing LIVEKIT_URL or credentials"
        return jsonify(diagnostics), 200

    try:
        # Try minting a token to ensure credentials parse correctly
        token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, identity=f"diag-{secrets.token_hex(3)}")
        token.add_grant(VideoGrant(room=ROOM_NAME, room_join=True))
        _ = token.to_jwt()
        diagnostics["token"]["ok"] = True
    except Exception as exc:  # pragma: no cover
        diagnostics["token"]["ok"] = False
        diagnostics["token"]["error"] = str(exc)

    return jsonify(diagnostics), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
