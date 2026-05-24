# Base image
FROM python:3.10-slim

# Intel Image Classification - ResNet50
# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir \
    flask pillow \
    torch==2.6.0+cpu \
    torchvision==0.21.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
