"""
POST /api/upload — Parse an uploaded file and return chunks.
Vercel serverless function.

The client sends a file via multipart form upload. This function parses it
(PDF/CSV/XLSX) and returns the text chunks as JSON. The client stores these
chunks in memory and sends them with subsequent /api/ask requests.
"""
from http.server import BaseHTTPRequestHandler
import json
import cgi
import io


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Parse multipart form data
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._error(400, "Expected multipart/form-data")
                return

            # Parse the multipart data
            environ = {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            }
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ=environ,
            )

            file_item = form["file"]
            if not file_item.filename:
                self._error(400, "No file uploaded")
                return

            filename = file_item.filename
            file_bytes = file_item.file.read()

            # Import here to avoid cold-start penalty on non-upload routes
            from api._lib.ingestion import ingest_file

            chunks = ingest_file(filename, file_bytes)

            self._json_response(200, {
                "ok": True,
                "filename": filename,
                "chunks": chunks,
                "total_chunks": len(chunks),
            })

        except ValueError as e:
            self._error(400, str(e))
        except Exception as e:
            self._error(500, f"Failed to parse file: {e}")

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
