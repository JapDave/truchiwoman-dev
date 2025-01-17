# syntax=docker/dockerfile:1

# Base image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the source code into the container
COPY . .

# Navigate into the 'code' folder as the working directory
WORKDIR /app/code

# Install dependencies from requirements.txt
# Make sure requirements.txt is at the root or inside 'code' (adjust path accordingly)
RUN python -m pip install --no-cache-dir -r requirements.txt

# Expose port 8080
EXPOSE 5000

# Run the application
CMD ["python", "main.py"]
