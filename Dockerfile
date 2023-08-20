# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

FROM eclipse-temurin:11
RUN mkdir /opt/app
COPY japp.jar /opt/app

# Expose port 8501
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]
