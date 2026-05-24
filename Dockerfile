# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install from local packages folder (no internet needed!)
RUN pip install --no-cache-dir --no-index \
    --find-links=/app/packages \
    flask pillow torch torchvision

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]