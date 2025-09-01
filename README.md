# AI Portrait Generator

## 📁 Structure du projet

```
AI_portrait/
├── README.md                    # Ce fichier
├── docker-compose.yml          # Configuration des services
├── .gitignore                  # Fichiers à ignorer
├── backend/
└── frontend/


```

## 🚀 Démarrage rapide

### Prérequis

- Docker et Docker Compose installés
- NVIDIA GPU avec drivers (pour l'accélération GPU)
- NVIDIA Container Toolkit installé
- Au moins 8 Go de VRAM GPU recommandés

### Installation automatique

```bash
# Cloner le projet
git clone <votre-repo>
cd AI_portrait

# Rendre les scripts exécutables
chmod +x scripts/*.sh

# Installation complète
./scripts/setup.sh
```

### Démarrage manuel

```bash
# 1. Nettoyer la mémoire GPU (recommandé)
./scripts/cleanup_gpu.sh

# 2. Démarrer les services
docker-compose up --build

# 3. Accéder à l'application
# Frontend: http://localhost:9000
# Backend API: http://localhost:8000
# Documentation API: http://localhost:8000/docs
```

## 🐳 Services Docker

### Backend (FastAPI + AI)
- **Port:** 8000
- **Technologies:** Python, FastAPI, PyTorch, Diffusers
- **GPU:** Utilise NVIDIA CUDA pour l'accélération
- **Endpoints:**
  - `GET /` - Status de l'API
  - `POST /generate` - Génération d'image
  - `GET /memory-status` - État mémoire GPU

### Frontend (Application web)
- **Port:** 9000
- **Technologies:** Vue.js/React, Quasar Framework
- **Fonctionnalités:**
  - Interface utilisateur pour upload d'images
  - Prévisualisation en temps réel
  - Téléchargement des résultats

## ⚙️ Configuration

### Variables d'environnement

Copiez `.env.example` vers `.env` et modifiez selon vos besoins :

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model Configuration
MODEL_PATH=/app/models
MODEL_NAME=stable-diffusion-xl

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Frontend Configuration
FRONTEND_PORT=9000
API_URL=http://localhost:8000
```

## 📦 Déploiement

### Production avec Docker

```bash
# Build des images de production
docker-compose -f docker-compose.prod.yml build

# Démarrage en production
docker-compose -f docker-compose.prod.yml up -d
```


## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

**Note:** Ce projet nécessite une GPU NVIDIA avec au moins 12 Go de VRAM pour fonctionner de manière optimale.
