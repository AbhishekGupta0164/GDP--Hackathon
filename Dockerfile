# ============================================================
# SafetyGuard X — Dockerfile
# HuggingFace Spaces compatible (port 7860)
# Build:  docker build -t safetyguard-x .
# Run:    docker run -p 7860:7860 safetyguard-x
# ============================================================

FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required for HF Spaces)
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app


# Install Python deps
COPY --chown=user requirements.txt .
RUN pip install --upgrade pip --no-cache-dir
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY --chown=user . .

# HF Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]