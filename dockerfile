# Utiliser une image Python légère comme base
FROM python:3.8-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le script Python et les fichiers nécessaires dans le conteneur
COPY weather-classification-TP.py /app/
COPY ResNet152V2-Weather-Classification-03.h5 /app/
COPY requirements.txt /app/

# Installer les dépendances système requises
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip et installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Créer les dossiers nécessaires
RUN mkdir -p /app/output

# Définir le point d'entrée pour exécuter le script avec des arguments
ENTRYPOINT ["python", "weather-classification-TP.py"]
