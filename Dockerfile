# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# syntax=docker/dockerfile:1
FROM alpine:3.16.0
RUN apk add --no-cache java-cacerts openjdk17-jdk

# Expose port 8501
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]
