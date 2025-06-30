FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Generate model weights if they don't exist
RUN python generate_weights.py

# Expose the Flask port
EXPOSE 5000

# Run the application with gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "main:app"]
