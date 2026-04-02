# Use a slim Python image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code and your service account key
COPY . .

# Run the script
CMD ["python", "main.py"]