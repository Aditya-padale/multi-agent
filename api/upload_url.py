"""
POST /api/upload-url — Fetch and parse a webpage, return chunks.
Vercel serverless function (note: Vercel maps upload_url.py to /api/upload_url
but we configure vercel.json to route /api/upload-url to this function).
"""
from http.server import BaseHTTPRequestHandler
import json


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))
            url = body.get("url", "").strip()

            if not url:
                self._error(400, "URL is required")
                return

            from api._lib.ingestion import ingest_webpage

            chunks = ingest_webpage(url)

            self._json_response(200, {
                "ok": True,
                "url": url,
                "chunks": chunks,
                "total_chunks": len(chunks),
            })

        except Exception as e:
            self._error(400, f"Failed to fetch/parse URL: {e}")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json_response(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, detail):
        self._json_response(status, {"detail": detail})
