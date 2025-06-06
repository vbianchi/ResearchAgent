# Use an official Python runtime as a parent image
# Using slim-bullseye for a smaller footprint - Explicitly 3.12
# Match casing for FROM and AS
FROM python:3.12-slim-bullseye AS base

# Set environment variables using key=value format
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set timezone based on user's location - Optional but can be helpful for logs
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set the working directory in the container
WORKDIR /app

# <<< --- START: Install curl and other system dependencies --- >>>
# Update package lists and install curl. Other build dependencies removed as they are no longer needed.
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* # Clean up apt cache
# <<< --- END: Install system dependencies --- >>>


# Install uv (faster package installer) - Recommended based on README
# We use pip to bootstrap uv itself
# Use python3 explicitly for pip command
RUN python3 -m pip install uv

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .
# Install dependencies using uv (system-wide within the container)
# Using --system as recommended by uv for Docker base images
# Use python3 explicitly to invoke uv module
RUN python3 -m uv pip install --system -r requirements.txt

# Copy the rest of the application code into the container
# Note: For development, this will be overlaid by the volume mount in docker-compose.yml
COPY . .

# Expose ports used by the application (WebSocket and File Server)
EXPOSE 8765
EXPOSE 8766

# Define the command to run the application
# Use python3 explicitly
CMD ["python3", "-m", "backend.server"]