# Utiliser une image Python légère comme base
FROM python:3.8-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copying the python script and all the dependencies 
COPY weather-classification-TP.py /app/
COPY ResNet152V2-Weather-Classification-03.h5 /app/
COPY requirements.txt /app/

# Instaling system prerequises 
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# installing all the python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# create the output folder 
RUN mkdir -p /app/output

# this is the entry point for the python script 
ENTRYPOINT ["python", "weather-classification-TP.py"]
