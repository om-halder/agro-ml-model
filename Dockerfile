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

# Expose port
EXPOSE 5001

# Start the app
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "app:app"]
