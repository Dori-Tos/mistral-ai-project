# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /historic-fact-checker

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port if you have a web service (adjust as needed)
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application (adjust to your main file)
CMD ["python", "app/main.py"]