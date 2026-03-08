"""
Run EXPoly viz with real-time grain switch: FastAPI app with POST /api/export and Dash mounted at /.
Usage: uvicorn web_app.server:app --reload
From project root: python -m uvicorn web_app.server:app --host 0.0.0.0 --port 8050
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"

# Load Dash app
sys.path.insert(0, str(REPO_ROOT))
from web_app.app import app as dash_app

# Load run_export from scripts
spec = importlib.util.spec_from_file_location(
    "export_script",
    REPO_ROOT / "scripts" / "export_legacy_steps_for_web.py",
)
export_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_mod)
run_export = export_mod.run_export

app = FastAPI(title="EXPoly pipeline visualization")


class ExportBody(BaseModel):
    grain_id: int


@app.post("/api/export")
def api_export(body: ExportBody):
    """Run pipeline for the given grain_id and write data to web_app/data/."""
    dream3d_path_file = DATA_DIR / "dream3d_path.txt"
    if not dream3d_path_file.exists():
        raise HTTPException(
            status_code=400,
            detail="Run export script once first: python scripts/export_legacy_steps_for_web.py --dream3d <path> --out-dir web_app/data",
        )
    dream3d_path = Path(dream3d_path_file.read_text().strip())
    if not dream3d_path.exists():
        raise HTTPException(status_code=400, detail=f"Dream3D file not found: {dream3d_path}")
    try:
        run_export(dream3d_path, body.grain_id, DATA_DIR)
        return {"ok": True, "grain_id": body.grain_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", WSGIMiddleware(dash_app.server))


def _find_free_port(start: int = 8050, max_tries: int = 10) -> int:
    import socket
    for i in range(max_tries):
        port = start + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    return start


if __name__ == "__main__":
    import uvicorn
    port = _find_free_port(8050)
    if port != 8050:
        print(f"Port 8050 in use; using port {port}")
    print(f"Open http://127.0.0.1:{port}/ or http://127.0.0.1:{port}/pipeline or http://127.0.0.1:{port}/scale")
    uvicorn.run("web_app.server:app", host="127.0.0.1", port=port)
