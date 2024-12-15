# Use an official Python runtime as a parent image
FROM python:3.12.3-slim

# Set working directory in the container
WORKDIR /app

# Copy application files into the container
COPY . /app

# Update apt, install venv, and clean up
# Install necessary packages for PyAudio
RUN apt-get update && \
    apt-get install -y python3-venv python3-dev portaudio19-dev libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment in /opt/venv and add it to the PATH
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip

# Install dependencies from requirements.txt (excluding torch)
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install the CPU-only version of PyTorch
RUN pip install torch==2.2.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Set environment variable for PORT (default to 8080 if not provided)
ENV PORT="8080"

# Expose the port the app runs on
EXPOSE 8080

# Run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
