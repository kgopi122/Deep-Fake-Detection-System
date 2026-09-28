FROM python:3.10-slim
WORKDIR /app

# Install system dependencies required for OpenCV and PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to handle modern wheel metadata cleanly
RUN pip install --no-cache-dir --upgrade pip

# Use --extra-index-url so pip falls back to PyPI for standard build packages
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]