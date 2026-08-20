# Use an official Python runtime as a parent image
# Note: TensorFlow 2.21.0 ships wheels for cp310-cp313 only, so we use 3.13
FROM python:3.13.9-slim-bookworm

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run gunicorn when the container launches
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
