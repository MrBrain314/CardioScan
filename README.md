# 🫀 CardioScan

> Tableau de bord interactif d'analyse et de prédiction des maladies cardiaques

🔗 **Application déployée** : [cardioscan-ytwkxq5bhd4o8pweqmyd2u.streamlit.app](https://cardioscan-ytwkxq5bhd4o8pweqmyd2u.streamlit.app/)

---

## 📋 Aperçu

**CardioScan** permet d'explorer, nettoyer et analyser les données médicales de **521 patients** afin de prédire la présence d'une maladie cardiaque à l'aide d'un algorithme de classification **KNN** (K plus proches voisins).

---

## ✨ Fonctionnalités

| Page | Description |
|------|-------------|
| 📊 **Vue d'ensemble** | Statistiques descriptives, distribution des variables, matrice de corrélation |
| 🧹 **Nettoyage des données** | Détection des doublons, correction des outliers, imputation des valeurs manquantes |
| 📈 **Analyse comparative** | Groupby, tableaux croisés, catégorisation par tranche d'âge |
| 🤖 **Modèle prédictif** | KNN avec matrice de confusion, rapport de classification, courbe accuracy vs k |
| 🔮 **Simulateur patient** | Prédiction en temps réel à partir des données d'un nouveau patient |

---

## 📁 Structure du projet

```
CardioScan/
├── app.py                   # Application Streamlit principale
├── data/
│   └── heart_disease.csv    # Dataset (521 patients, 14 variables)
├── scripts/
│   └── generate_data.py     # Script de génération des données synthétiques
├── requirements.txt         # Dépendances Python
└── README.md
```

---

## 🗃️ Dataset

Le fichier `heart_disease.csv` contient des données médicales sur **521 patients** avec **14 variables** :

| Variable | Description |
|----------|-------------|
| `age` | Âge du patient (années) |
| `sex` | Sexe (0 = Homme, 1 = Femme) |
| `cp` | Type de douleur thoracique (0 à 3) |
| `trestbps` | Pression artérielle au repos (mmHg) |
| `chol` | Cholestérol sérique (mg/dL) |
| `fbs` | Glycémie à jeun > 120 mg/dL (0/1) |
| `restecg` | ECG au repos (0, 1, 2) |
| `thalach` | Fréquence cardiaque maximale (bpm) |
| `exang` | Angine à l'effort (0/1) |
| `oldpeak` | Dépression du segment ST |
| `slope` | Pente du segment ST (0, 1, 2) |
| `ca` | Nombre de vaisseaux colorés (0 à 3) |
| `thal` | Thalassémie (1, 2, 3) |
| `target` | Maladie cardiaque : **1 = Malade**, **0 = Sain** |

---

## 🚀 Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-utilisateur/CardioScan.git
cd CardioScan

# Créer et activer l'environnement virtuel
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac / Linux

# Installer les dépendances
pip install -r requirements.txt
```

---

## ▶️ Lancement

```bash
streamlit run app.py
```

---

## 🛠️ Technologies utilisées

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.x-blue?logo=pandas)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-green)

---

> 📝 Projet réalisé dans un cadre éducatif. Les données sont synthétiques et générées à partir des distributions du **Cleveland Heart Disease Dataset**.
