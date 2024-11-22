FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

COPY weather-classification-TP.py /app/
COPY ResNet152V2-Weather-Classification-03.h5 /app/
COPY requirements.txt /app/

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Entry point for the container
ENTRYPOINT ["python", "weather-classification-TP.py"]

