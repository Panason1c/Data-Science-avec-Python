# Détection de Fraude Bancaire — Projet Data Science

> Modèle de Machine Learning pour détecter automatiquement les transactions bancaires frauduleuses, exposé via une API FastAPI.

---

## Structure du projet

```
projet_finale/
├── EDA.ipynb                 # Analyse exploratoire des données
├── preprocessing.ipynb       # Preprocessing + modèles + sauvegarde
├── fraudTrain.csv            # Dataset (à télécharger sur Kaggle)
└── api/
    ├── main.py               # API FastAPI
    ├── model.pkl             # Modèle entraîné (généré par le notebook)
    └── requirements.txt      # Dépendances Python
```

---

## Dataset

Télécharger le fichier **fraudTrain.csv** depuis Kaggle :

🔗 [Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data?select=fraudTrain.csv)

Placer le fichier `fraudTrain.csv` au même niveau que `EDA.ipynb`.

---

## Installation & Lancement

### Étape 1 — Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 2 — Générer le modèle

Ouvrir et exécuter toutes les cellules de `preprocessing.ipynb`.  
Le fichier `model.pkl` sera automatiquement sauvegardé dans le dossier `api/`.

### Étape 3 — Lancer l'API

```bash
cd api
pip install uvicorn --user
python -m uvicorn main:app --reload
```

### Étape 4 — Accéder à l'interface

Ouvrir le navigateur sur :

```
http://127.0.0.1:8000/docs
```

---

## Utilisation de l'API

### Endpoint disponible

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Vérifie que l'API tourne |
| `GET` | `/health` | Statut de l'API |
| `POST` | `/predict` | Prédit si une transaction est frauduleuse |

### Exemple de requête

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "trans_date_trans_time": "2019-06-21 22:15:00",
    "category": "shopping_net",
    "amt": 349.99,
    "gender": "M",
    "lat": 36.0788,
    "long": -81.1781,
    "city_pop": 3495,
    "job": "Psychologist, counselling",
    "dob": "1988-03-09",
    "merch_lat": 36.011293,
    "merch_long": -82.048315
  }'
```

### Exemple de réponse

```json
{
  "is_fraud": 0,
  "fraud_probability": 0.0312,
  "risk_level": "FAIBLE",
  "transaction_amount": 349.99,
  "category": "shopping_net",
  "message": "✓ Transaction légitime"
}
```

### Niveaux de risque

| Probabilité | Niveau |
|---|---|
| < 30% | FAIBLE |
| 30% — 70% | MOYEN |
| > 70% | ÉLEVÉ |

---

## Modèles utilisés

| Type | Algorithme | Justification |
|---|---|---|
| Supervisé | Random Forest | Robuste, non-linéaire, importance des features |
| Non supervisé | K-Means (k=4) | Segmentation comportementale des transactions |


---

## Technologies

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![pandas](https://img.shields.io/badge/pandas-2.2-purple)
