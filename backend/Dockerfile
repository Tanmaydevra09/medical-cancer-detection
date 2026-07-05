FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (e.g., for libGL used by OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (frontend and models should ideally be handled via .dockerignore or separately, but we'll copy the root)
# The context will be the root directory
COPY . /app/

# Environment Variables
ENV IMAGE_MODEL_PYTHON=python
ENV LUNG_MODEL_PYTHON=python
ENV BLOOD_PYTHON=python
ENV LOAD_SKLEARN_LIBOMP=0
ENV FLASK_APP=backend/app.py

EXPOSE 5000

# Start Gunicorn server
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "backend.app:app"]
