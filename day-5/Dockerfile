# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY .env .

# Set PATH for user-installed packages
ENV PATH=/root/.local/bin:$PATH

# Create data directory
RUN mkdir -p /app/data

# Default command
CMD ["python", "-m", "src.file_analyzer"]