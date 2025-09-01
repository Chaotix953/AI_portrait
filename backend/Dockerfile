# Utilise une image Python officielle
FROM python:3.8-slim


# Définit le répertoire de travail
WORKDIR /app

# Copie les fichiers nécessaires
COPY requirements.txt .
COPY api.py .
COPY results/ .
# COPY images/ ./images

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

RUN pip install --upgrade pip

#installer pytorch avec cuda
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121

# Installe les dépendances
RUN pip install -r requirements.txt

# Expose le port utilisé par Uvicorn
EXPOSE 8000

# Commande pour lancer l'API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]



