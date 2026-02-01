FROM python:3.10-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies (FIXES libxcb error)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libx11-xcb1 \
    libxcb1 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Railway assigns PORT dynamically
EXPOSE 8080

CMD gunicorn app:app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT

