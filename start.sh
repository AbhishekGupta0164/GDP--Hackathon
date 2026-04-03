# ============================================================
# SafetyGuard X — Startup Script
# Clean startup script
# !/bin/bash
# ============================================================
echo "Starting SafetyGuard X..."
echo "Port: ${PORT:-7860}"


python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-7860} \
    --workers 1 \
    --log-level info
```
