"""
Génération de 521 patients synthétiques réalistes
basée sur les distributions du Cleveland Heart Disease Dataset.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 521

# ── Target (variable cible) ──────────────────────────────────────────────────
# ~54 % malades, ~46 % sains
target = np.random.choice([0, 1], size=N, p=[0.46, 0.54])

# ── Âge ──────────────────────────────────────────────────────────────────────
# Malades : légèrement plus âgés
age = np.where(
    target == 1,
    np.clip(np.random.normal(56, 9, N).astype(int), 35, 77),
    np.clip(np.random.normal(52, 9, N).astype(int), 29, 74)
)

# ── Sexe ─────────────────────────────────────────────────────────────────────
# ~68 % hommes dans le dataset original
sex = np.random.choice(["Male", "Female"], size=N, p=[0.68, 0.32])

# ── cp : type de douleur thoracique (0-3) ────────────────────────────────────
# Malades : cp=0 (asymptomatique) plus fréquent
cp = np.where(
    target == 1,
    np.random.choice([0, 1, 2, 3], size=N, p=[0.55, 0.12, 0.23, 0.10]),
    np.random.choice([0, 1, 2, 3], size=N, p=[0.30, 0.22, 0.33, 0.15])
)

# ── trestbps : pression artérielle au repos ──────────────────────────────────
trestbps = np.clip(np.random.normal(132, 17, N).astype(int), 90, 200)

# ── chol : cholestérol sérique ───────────────────────────────────────────────
chol = np.clip(np.random.normal(246, 51, N).astype(int), 140, 420)

# ── fbs : glycémie à jeun > 120 (0/1) ───────────────────────────────────────
fbs = np.random.choice([0, 1], size=N, p=[0.85, 0.15])

# ── restecg : ECG au repos (0, 1, 2) ────────────────────────────────────────
restecg = np.random.choice([0, 1, 2], size=N, p=[0.50, 0.48, 0.02])

# ── thalach : fréquence cardiaque maximale ───────────────────────────────────
# Malades : thalach plus bas
thalach = np.where(
    target == 1,
    np.clip(np.random.normal(139, 22, N).astype(int), 80, 195),
    np.clip(np.random.normal(163, 18, N).astype(int), 110, 202)
)

# ── exang : angine à l'effort (0/1) ─────────────────────────────────────────
# Malades : exang=1 plus fréquent
exang = np.where(
    target == 1,
    np.random.choice([0, 1], size=N, p=[0.45, 0.55]),
    np.random.choice([0, 1], size=N, p=[0.80, 0.20])
)

# ── oldpeak : dépression ST ──────────────────────────────────────────────────
# Malades : oldpeak plus élevé
oldpeak = np.where(
    target == 1,
    np.clip(np.random.exponential(1.8, N), 0.0, 6.2).round(1),
    np.clip(np.random.exponential(0.6, N), 0.0, 4.0).round(1)
)

# ── slope : pente du segment ST (0-2) ───────────────────────────────────────
slope = np.where(
    target == 1,
    np.random.choice([0, 1, 2], size=N, p=[0.15, 0.55, 0.30]),
    np.random.choice([0, 1, 2], size=N, p=[0.05, 0.35, 0.60])
)

# ── ca : nombre de vaisseaux colorés (0-3) ───────────────────────────────────
# Malades : ca plus élevé
ca = np.where(
    target == 1,
    np.random.choice([0, 1, 2, 3], size=N, p=[0.35, 0.28, 0.22, 0.15]),
    np.random.choice([0, 1, 2, 3], size=N, p=[0.72, 0.16, 0.08, 0.04])
)

# ── thal : thalassémie (1-3) ──────────────────────────────────────────────────
# Malades : thal=3 (défaut réversible) plus fréquent
thal = np.where(
    target == 1,
    np.random.choice([1, 2, 3], size=N, p=[0.05, 0.35, 0.60]),
    np.random.choice([1, 2, 3], size=N, p=[0.07, 0.72, 0.21])
)

# ── Identifiants patients ────────────────────────────────────────────────────
ids = np.arange(1001, 1001 + N)

# ── Assemblage du DataFrame ──────────────────────────────────────────────────
df = pd.DataFrame({
    "ID"      : ids,
    "age"     : age,
    "sex"     : sex,
    "cp"      : cp,
    "trestbps": trestbps,
    "chol"    : chol,
    "fbs"     : fbs,
    "restecg" : restecg,
    "thalach" : thalach,
    "exang"   : exang,
    "oldpeak" : oldpeak,
    "slope"   : slope,
    "ca"      : ca,
    "thal"    : thal,
    "target"  : target
}).set_index("ID")

# ── Injection des imperfections réalistes ────────────────────────────────────

# 1. Valeurs manquantes (~3-5 % par colonne concernée)
colonnes_manquantes = {
    "thalach" : 0.03,   # fréquence cardiaque parfois non mesurée
    "ca"      : 0.04,   # nombre de vaisseaux parfois absent
    "thal"    : 0.03,   # thalassémie parfois non renseignée
    "chol"    : 0.02,   # cholestérol parfois manquant
    "restecg" : 0.02,   # ECG parfois non réalisé
    "target"  : 0.01,   # quelques patients non diagnostiqués
}
for col, taux in colonnes_manquantes.items():
    n_manquants = int(N * taux)
    idx_manquants = np.random.choice(df.index, size=n_manquants, replace=False)
    df.loc[idx_manquants, col] = np.nan

# 2. Valeurs aberrantes (outliers)
# thalach : 3 valeurs avec virgule mal placée (ex: 1650 au lieu de 165)
idx_outliers_thalach = np.random.choice(df[df["thalach"].notna()].index, size=3, replace=False)
df.loc[idx_outliers_thalach, "thalach"] = df.loc[idx_outliers_thalach, "thalach"] * 10

# trestbps : 2 valeurs extrêmes
idx_outliers_bps = np.random.choice(df.index, size=2, replace=False)
df.loc[idx_outliers_bps, "trestbps"] = np.random.choice([15, 320], size=2)

# chol : 1 valeur aberrante
idx_outlier_chol = np.random.choice(df[df["chol"].notna()].index, size=1)
df.loc[idx_outlier_chol, "chol"] = 999

# 3. Rapport final
total_manquants = df.isnull().sum().sum()
total_outliers  = 6  # 3 thalach + 2 trestbps + 1 chol

# ── Export CSV ───────────────────────────────────────────────────────────────
df.to_csv("../data/heart_disease.csv", sep=";")
print(f"Fichier genere : heart_disease_521.csv ({len(df)} lignes)")
print(f"\nRepartition target (avec NaN) :\n{df['target'].value_counts(dropna=False)}")
print(f"\nValeurs manquantes par colonne :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nTotal valeurs manquantes : {total_manquants}")
print(f"Total outliers injectes  : {total_outliers}")
print(f"\nApercu :\n{df.head()}")
