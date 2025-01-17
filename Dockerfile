# syntax=docker/dockerfile:1

# Base image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the source code into the container
COPY . .

# Install dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Expose port 8080
EXPOSE 5000

# Run the application (main.py is in the 'code' folder)
CMD ["python", "code/main.py"]
