# Use the official Python image with Debian (includes Python)
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenJDK 11
RUN apt-get update && apt-get install -y openjdk-11-jre

# Expose port 8501
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]
