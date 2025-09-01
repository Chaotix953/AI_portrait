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

- Docker et docker-compose installés
- NVIDIA GPU avec drivers (pour l'accélération GPU)
- Au moins 12 Go de VRAM GPU recommandés

### 📦 Déploiement

```bash
# Cloner le projet
git clone <votre-repo>
cd AI_portrait

#lancer les conteneurs 
docker-compose up --build
```

## 🐳 Services Docker

### Backend (FastAPI + AI)
- **Port:** 8000
- **Technologies:** Python, FastAPI, PyTorch, Diffusers
- **GPU:** Utilise NVIDIA CUDA pour l'accélération
- **Endpoints:**
  - `GET /` - Status de l'API
  - `POST /generate` - Génération d'image

### Frontend (Application web)
- **Port:** 9000
- **Technologies:** Vue.js/React, Quasar Framework
- **Fonctionnalités:**
  - Interface utilisateur pour upload d'images
  - Prévisualisation en temps réel
  - Téléchargement des résultats

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

**Note:** Ce projet nécessite une GPU NVIDIA avec au moins 12 Go de VRAM pour fonctionner de manière optimale.
