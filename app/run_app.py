"""
Application Launcher for Federated GAT-IDS Full-Stack Web Application.
Launches Uvicorn server hosting FastAPI backend and single-page dashboard UI.
"""

import sys
import os
from pathlib import Path

# Add project root directory to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def check_dependencies():
    """Verify required libraries are installed."""
    missing = []
    for pkg in ["fastapi", "uvicorn", "pydantic", "torch", "pandas", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"[!] Warning: Missing packages: {missing}")
        print("    Install using: pip install -r app/requirements_app.txt")

def main():
    check_dependencies()
    import uvicorn

    print("=" * 70)
    print("  Federated Graph Attention Network (GAT) Intrusion Detection System")
    print("  Full-Stack Production Web Application")
    print("=" * 70)
    print("  [+] Server URL:       http://127.0.0.1:8000")
    print("  [+] REST API Docs:    http://127.0.0.1:8000/docs")
    print("  [+] Metrics Endpoint: http://127.0.0.1:8000/api/metrics")
    print("=" * 70)
    print("Press Ctrl+C to stop the server.\n")

    uvicorn.run(
        "app.backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    main()
