# 1. Base image: official lightweight Python, same version used to train the model
FROM python:3.13-slim

# 2. Working directory inside the container (created automatically)
WORKDIR /app

# 3. Copy ONLY the dependency list first, to use Docker layer caching
COPY requirements.txt .

# 4. Install the pinned dependencies (--no-cache-dir keeps the image smaller)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the trained model and the prediction application
COPY model.joblib .
COPY app.py .

# 6. Default command when a container starts from this image
CMD ["python", "app.py"]