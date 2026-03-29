# Multi-stage build
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ src/
COPY .env .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

RUN mkdir -p output/plots output/logs

CMD ["python", "-m", "src.pipeline"]