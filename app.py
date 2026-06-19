import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(
    page_title="CardioScan",
    page_icon="🫀",
    layout="wide"
)

st.markdown("""
<style>
/* ── Fond général ── */
[data-testid="stAppViewContainer"] {
    background-color: #f0f2f5;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #2d6a4f !important;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #d4ede5 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem !important;
    padding: 6px 0 !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] > div {
    gap: 4px;
}
/* Radio sélectionné */
[data-testid="stSidebar"] [aria-checked="true"] + div p {
    color: #52b788 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] hr { border-color: #2d6a4f !important; }

/* ── Titres de page ── */
h1 { color: #2d6a4f !important; font-weight: 800 !important; }
h2, h3 { color: #2d6a4f !important; font-weight: 700 !important; }

/* ── Métriques Streamlit ── */
[data-testid="stMetric"] {
    background: white;
    border-radius: 14px;
    padding: 18px 20px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-top: 4px solid #52b788;
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #2d6a4f !important; font-size: 2rem !important; font-weight: 800 !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: white;
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 10px;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* ── Bouton ── */
[data-testid="stButton"] > button {
    background: #2d6a4f !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 12px rgba(45,106,79,0.3);
    transition: all 0.2s;
}
[data-testid="stButton"] > button:hover {
    background: #2d6a4f !important;
    box-shadow: 0 6px 16px rgba(45,106,79,0.4);
}

/* ── Selectbox / Slider ── */
[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: #d1fae5 !important;
    background: white !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #2d6a4f !important;
}

/* ── Graphiques matplotlib ── */
[data-testid="stImage"] img { border-radius: 12px; }

/* ── Tabs / Sections ── */
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }

/* ── Succès / Warnings ── */
[data-testid="stAlert"] { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Chargement et préparation ───

@st.cache_data
def load_raw():
    return pd.read_csv("data/heart_disease.csv", sep=";", index_col=0)

@st.cache_data
def load_and_prepare():
    df = pd.read_csv("data/heart_disease.csv", sep=";", index_col=0)

    # 1. Doublons
    df = df.drop_duplicates(keep="first")

    # 2. Encodage sex : Male → 0, Female → 1
    df["sex"] = df["sex"].replace({"Male": 0, "Female": 1})

    # 3. Correction outliers thalach (hors [50, 250])
    mediane_valide = df.loc[df["thalach"].between(50, 250), "thalach"].median()
    df.loc[~df["thalach"].between(50, 250) & df["thalach"].notna(), "thalach"] = mediane_valide

    # 4. Suppression des lignes sans target
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    # 5. Imputation par le mode : ca, exang
    for col in ["ca", "exang"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 6. Imputation par la médiane : reste
    df = df.fillna(df.median(numeric_only=True))

    # 7. Catégories d'âge avec pd.cut
    df["age_categ"] = pd.cut(
        df["age"],
        bins=[0, 40, 60, 100],
        labels=["jeune", "adulte", "senior"],
        right=True
    )

    return df

df_raw   = load_raw()
df_clean = load_and_prepare()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div style="padding:10px 0 20px 0;">
  <div style="font-size:1.6rem;font-weight:900;color:#52b788;letter-spacing:-0.5px;">🫀 CardioScan</div>
  <div style="font-size:0.78rem;color:#8ab9a8;margin-top:2px;">Analyse des maladies cardiaques</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div style='color:#8ab9a8;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;'>Menu</div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["📊 Vue d'ensemble", "🧹 Nettoyage des données", "📈 Analyse comparative", "🤖 Modèle prédictif", "🔮 Simulateur patient"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='color:#8ab9a8;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;'>Dataset</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style="background:#1e4d3a;border-radius:12px;padding:14px 16px;margin-top:4px;">
  <div style="color:#52b788;font-weight:700;font-size:0.88rem;">heart_disease.csv</div>
  <div style="color:#8ab9a8;font-size:0.8rem;margin-top:8px;">👥 {len(df_clean)} patients</div>
  <div style="color:#8ab9a8;font-size:0.8rem;">📋 14 variables</div>
  <div style="margin-top:10px;background:#2d6a4f;border-radius:8px;padding:8px 10px;">
    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
      <span style="color:#e05c5c;font-size:0.82rem;">● Malades</span>
      <span style="color:#e05c5c;font-weight:700;font-size:0.82rem;">{int(df_clean['target'].sum())}</span>
    </div>
    <div style="display:flex;justify-content:space-between;">
      <span style="color:#52b788;font-size:0.82rem;">● Sains</span>
      <span style="color:#52b788;font-weight:700;font-size:0.82rem;">{int((df_clean['target']==0).sum())}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 - VUE D'ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📊 Vue d'ensemble":
    st.title("📊 Vue d'ensemble")

    st.markdown("""
    <div style="background:white;border-radius:14px;padding:20px 24px;margin-bottom:24px;
                box-shadow:0 2px 12px rgba(0,0,0,0.06);border-left:5px solid #52b788;">
        <div style="font-size:1.05rem;color:#374151;line-height:1.7;">
            Ce tableau de bord analyse les données médicales de <strong>516 patients</strong>
            dans le but de comprendre et prédire la présence d'une <strong>maladie cardiaque</strong>.<br><br>
            Chaque patient est décrit par <strong>13 variables médicales</strong> (âge, pression artérielle,
            fréquence cardiaque, résultats d'examens...) et une variable cible <code>target</code>
            indiquant s'il est <span style="color:#e05c5c;font-weight:700;">malade (1)</span>
            ou <span style="color:#2d6a4f;font-weight:700;">sain (0)</span>.<br><br>
            Explorez les différentes sections du menu pour analyser les données,
            comprendre les étapes de nettoyage, visualiser les tendances,
            et tester le modèle de prédiction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Patients",  len(df_clean))
    c2.metric("Variables", df_clean.shape[1] - 1)
    c3.metric("Malades",   int(df_clean["target"].sum()))
    c4.metric("Sains",     int((df_clean["target"] == 0).sum()))
    c5.metric("NaN bruts", int(df_raw.isna().sum().sum()))

    st.markdown("---")

    # ── Données brutes ──
    with st.expander("📂 Données brutes (521 lignes)", expanded=False):
        st.dataframe(df_raw, use_container_width=True)

    # ── Description des variables ──
    st.subheader("Description des variables")
    variables_info = pd.DataFrame([
        ("age",      "Age",                          "Numérique",    "Âge du patient en années"),
        ("sex",      "Sexe",                         "Catégorielle", "0 = Homme, 1 = Femme"),
        ("cp",       "Type de douleur thoracique",   "Catégorielle", "0 = Asymptomatique, 1 = Angine typique, 2 = Angine atypique, 3 = Autre"),
        ("trestbps", "Pression artérielle au repos", "Numérique",    "Mesurée en mmHg à l'admission"),
        ("chol",     "Cholestérol sérique",          "Numérique",    "Taux de cholestérol en mg/dL"),
        ("fbs",      "Glycémie à jeun",              "Catégorielle", "1 = Glycémie > 120 mg/dL, 0 = Normale"),
        ("restecg",  "ECG au repos",                 "Catégorielle", "0 = Normal, 1 = Anomalie ST-T, 2 = Hypertrophie ventriculaire"),
        ("thalach",  "Fréquence cardiaque maximale", "Numérique",    "Fréquence cardiaque max atteinte à l'effort (bpm)"),
        ("exang",    "Angine à l'effort",            "Catégorielle", "1 = Présente, 0 = Absente"),
        ("oldpeak",  "Dépression du segment ST",     "Numérique",    "Dépression ST induite par l'effort par rapport au repos"),
        ("slope",    "Pente du segment ST",          "Catégorielle", "0 = Descendante, 1 = Plate, 2 = Ascendante"),
        ("ca",       "Vaisseaux colorés (scintigraphie)", "Catégorielle", "Nombre de vaisseaux principaux colorés (0 à 3)"),
        ("thal",     "Thalassémie",                  "Catégorielle", "1 = Normal, 2 = Défaut fixe, 3 = Défaut réversible"),
        ("target",   "Maladie cardiaque (cible)",    "Catégorielle", "1 = Malade, 0 = Sain"),
    ], columns=["Variable", "Traduction", "Type", "Description"])
    st.dataframe(variables_info, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Stats descriptives ──
    st.subheader("Statistiques descriptives")
    cols_num = df_clean.select_dtypes(include="number").columns.drop("target")
    st.dataframe(df_clean[cols_num].describe().round(2), use_container_width=True)

    st.markdown("---")

    # ── Graphiques côte à côte ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Répartition - variable cible")
        counts = df_clean["target"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            counts, labels=["Malade (1)", "Sain (0)"],
            autopct="%1.1f%%", colors=["tomato", "steelblue"],
            startangle=90, wedgeprops={"edgecolor": "white"}
        )
        ax.set_title("Target (0 = Sain / 1 = Malade)")
        st.pyplot(fig); plt.close()
        st.markdown("""
        <div style="background:#f8f9fa;border-left:4px solid #e05c5c;border-radius:8px;
                    padding:14px 18px;margin-top:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
        📌 <b>Interprétation</b><br><br>
        Sur 516 patients, <b>283 sont malades (54,8 %)</b> et 233 sont sains (45,2 %).
        Il y a donc un peu plus de malades que de sains dans ce jeu de données.
        Cet écart est faible - les deux groupes sont bien représentés.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.subheader("Proportions par sexe")
        prop = pd.crosstab(df_clean["sex"], df_clean["target"], normalize="index")
        prop.index   = ["Homme (0)", "Femme (1)"]
        prop.columns = ["Sain", "Malade"]
        fig, ax = plt.subplots(figsize=(5, 4))
        prop.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="white")
        ax.set_xticklabels(prop.index, rotation=0)
        ax.set_ylabel("Proportion")
        ax.set_title("Sexe vs Maladie cardiaque")
        ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("""
        <div style="background:#f8f9fa;border-left:4px solid #e05c5c;border-radius:8px;
                    padding:14px 18px;margin-top:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
        📌 <b>Interprétation</b><br><br>
        <b>56 % des hommes sont malades</b>, contre <b>52 % des femmes</b>.
        Les hommes ont donc un risque légèrement plus élevé de maladie cardiaque.
        Cela dit, la différence entre les deux sexes reste faible dans ces données.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Distribution interactive ──
    st.subheader("Distribution d'une variable")
    col_select, _ = st.columns([2, 3])
    with col_select:
        variable = st.selectbox("Choisir une variable", list(cols_num))

    fig, ax = plt.subplots(figsize=(9, 3))
    for val, label, color in [(0, "Sain", "steelblue"), (1, "Malade", "tomato")]:
        data = df_clean[df_clean["target"] == val][variable].dropna()
        ax.hist(data, alpha=0.65, label=label, color=color, bins=20, edgecolor="white")
    ax.set_xlabel(variable)
    ax.set_ylabel("Nb patients")
    ax.set_title(f"Distribution de « {variable} » selon la maladie")
    ax.legend(); plt.tight_layout()
    st.pyplot(fig); plt.close()

    interpretations = {
        "age"     : "Les malades ont en moyenne <b>55 ans</b>, contre <b>52 ans</b> pour les sains. Plus on vieillit, plus le risque de maladie cardiaque augmente.",
        "thalach" : "Les malades atteignent en moyenne <b>153 bpm</b> à l'effort, contre <b>164 bpm</b> pour les sains. Un cœur malade se fatigue plus vite et monte moins haut en fréquence.",
        "chol"    : "Le taux de cholestérol est très similaire entre malades (~249 mg/dL) et sains (~253 mg/dL). Dans ces données, le cholestérol seul ne permet pas de distinguer les deux groupes.",
        "oldpeak" : "Les malades ont une dépression ST moyenne de <b>1,6</b>, contre seulement <b>0,5</b> pour les sains. Plus cette valeur est élevée, plus le cœur est en difficulté à l'effort.",
        "trestbps": "La pression artérielle est légèrement plus haute chez les malades, mais la différence est faible. Ce n'est pas un indicateur suffisant à lui seul.",
        "fbs"     : "La glycémie à jeun est quasi identique entre les deux groupes. Dans ce jeu de données, la glycémie ne distingue pas bien les malades des sains.",
        "restecg" : "Les résultats ECG au repos sont globalement similaires entre les deux groupes. Les anomalies sont un peu plus fréquentes chez les malades, mais l'écart reste faible.",
        "exang"   : "L'angine à l'effort (douleur dans la poitrine pendant un effort physique) est <b>beaucoup plus fréquente chez les malades</b>. C'est un signe fort de problème cardiaque.",
        "slope"   : "Les malades ont plus souvent une pente plate ou descendante du segment ST, ce qui indique que leur cœur réagit moins bien à l'effort.",
        "ca"      : "Les malades ont en général <b>plus d'artères obstruées</b> visibles à la scintigraphie. Plus ce nombre est élevé, plus le risque cardiaque est important.",
        "thal"    : "Les malades présentent plus souvent un résultat de type 3 (défaut réversible), ce qui signifie que certaines zones du cœur manquent de sang à l'effort mais récupèrent au repos - signe d'une maladie coronarienne.",
    }
    if variable in interpretations:
        st.markdown(f"""
        <div style="background:#f8f9fa;border-left:4px solid #5c8ee0;border-radius:8px;
                    padding:14px 18px;margin-top:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
        📌 <b>Interprétation - {variable}</b><br><br>
        {interpretations[variable]}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Matrice de corrélation (matplotlib) ──
    st.subheader("Matrice de corrélation")
    corr = df_clean[list(cols_num) + ["target"]].corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    ax.set_title("Corrélations entre variables")
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.markdown("""
    <div style="background:#f8f9fa;border-left:4px solid #e05c5c;border-radius:8px;
                padding:14px 18px;margin-top:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
    📌 <b>Interprétation</b><br><br>
    Les variables qui influencent le plus la maladie cardiaque sont :<br>
    - <b>thalach</b> : plus la fréquence cardiaque max est <i>basse</i>, plus le risque est élevé<br>
    - <b>oldpeak</b>, <b>ca</b>, <b>exang</b> : plus leurs valeurs sont <i>élevées</i>, plus le risque augmente<br><br>
    À l'inverse, <b>chol</b> (cholestérol) et <b>fbs</b> (glycémie) sont très peu liés à la maladie dans ces données.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 - NETTOYAGE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🧹 Nettoyage des données":
    st.title("🧹 Nettoyage des données")

    st.subheader("Données brutes")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.markdown("---")
    st.subheader("Étapes de nettoyage")

    # 2.1 Doublons
    with st.expander("2.1 - Doublons (`duplicated` / `drop_duplicates`)", expanded=True):
        nb = df_raw.duplicated().sum()
        if nb == 0:
            st.success(f"✅ Aucun doublon détecté ({nb})")
        else:
            st.warning(f"⚠️ {nb} doublon(s) supprimé(s)")

    # 2.2 Encodage
    with st.expander("2.2 - Encodage de `sex` (`replace`)", expanded=True):
        st.code("df['sex'] = df['sex'].replace({'Male': 0, 'Female': 1})")
        avant  = df_raw["sex"].value_counts().rename("Avant")
        apres  = df_clean["sex"].value_counts().rename("Après")
        st.dataframe(pd.concat([avant, apres], axis=1), use_container_width=True)

    # 2.3 Outliers
    with st.expander("2.3 - Outliers dans `thalach` (filtrage conditionnel)", expanded=True):
        thalach_num = pd.to_numeric(df_raw["thalach"], errors="coerce")
        outliers = df_raw[~thalach_num.between(50, 250) & thalach_num.notna()]
        if outliers.empty:
            st.success("✅ Aucun outlier")
        else:
            st.warning(f"⚠️ {len(outliers)} valeur(s) hors de [50, 250] bpm :")
            st.dataframe(outliers[["thalach"]], use_container_width=True)
            med = thalach_num[thalach_num.between(50, 250)].median()
            st.info(f"Remplacées par la médiane des valeurs valides : **{med} bpm**")

    # 2.4 NaN
    with st.expander("2.4 - Valeurs manquantes (`isna().sum()`)", expanded=True):
        nan_df = df_raw.isna().sum().reset_index()
        nan_df.columns = ["Colonne", "NaN"]
        nan_df = nan_df[nan_df["NaN"] > 0]
        st.dataframe(nan_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(nan_df["Colonne"], nan_df["NaN"], color="tomato", edgecolor="white")
        ax.set_title("Valeurs manquantes par colonne")
        ax.set_ylabel("Nb NaN")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # 2.5 Suppression target NaN
    with st.expander("2.5 - Suppression des lignes sans `target` (`dropna`)", expanded=True):
        n_av = len(df_raw)
        n_ap = len(df_clean)
        c1, c2, c3 = st.columns(3)
        c1.metric("Avant",     n_av)
        c2.metric("Après",     n_ap)
        c3.metric("Supprimées", n_av - n_ap)

    # 2.6 Imputation
    with st.expander("2.6 - Imputation mode & médiane (`fillna`)", expanded=True):
        st.markdown("- **`ca`, `exang`** → mode (variables qualitatives)")
        st.markdown("- **Autres colonnes** → médiane (variables quantitatives)")
        st.success(f"✅ {df_clean.isna().sum().sum()} valeur(s) manquante(s) après imputation")

    st.markdown("---")
    st.subheader("Données nettoyées")
    st.dataframe(df_clean.drop(columns=["age_categ"]).head(20), use_container_width=True)

    # Normalisation
    st.markdown("---")
    st.subheader("3.2 - Normalisation Min-Max → [-1, 1]")
    st.latex(r"X_{new} = 2\,\frac{X - X_{min}}{X_{max} - X_{min}} - 1")

    X     = df_clean.drop(columns=["target", "age_categ"])
    X_norm = 2 * (X - X.min()) / (X.max() - X.min()) - 1
    st.dataframe(X_norm.head(10).round(3), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Min global", f"{X_norm.min().min():.2f}")
    c2.metric("Max global", f"{X_norm.max().max():.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 - ANALYSE COMPARATIVE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Analyse comparative":
    st.title("📈 Analyse comparative")
    st.markdown("Cette page illustre les techniques d'agrégation apprises.")

    st.markdown("---")

    # ── Groupby + agg ──
    st.subheader("Statistiques par statut (sain / malade) - `groupby + agg`")

    fonctions = {
        "age"     : ["mean", "min", "max"],
        "thalach" : ["mean", "min", "max"],
        "chol"    : ["mean", "median"],
        "oldpeak" : ["mean", "median"],
        "trestbps": ["mean", "median"]
    }
    stats = df_clean.groupby("target").agg(fonctions).round(1)
    stats.index = ["Sains (0)", "Malades (1)"]
    st.dataframe(stats, use_container_width=True)
    st.markdown("""
    <div style="background:#f8f9fa;border-left:4px solid #e05c5c;border-radius:8px;
                padding:14px 18px;margin-top:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
    📌 <b>Interprétation</b><br><br>
    En comparant les deux groupes, on remarque plusieurs différences clés :<br>
    - <b>Âge</b> : les malades ont en moyenne 55 ans contre 52 ans pour les sains - ils sont légèrement plus âgés.<br>
    - <b>Fréquence cardiaque max (thalach)</b> : les malades atteignent 138 bpm en moyenne, contre 163 bpm pour les sains - leur cœur monte moins haut à l'effort.<br>
    - <b>Dépression ST (oldpeak)</b> : 1,6 chez les malades contre 0,5 chez les sains - signe que leur cœur souffre davantage pendant l'effort.<br>
    - <b>Cholestérol et pression artérielle</b> : très similaires entre les deux groupes, ces variables distinguent peu les malades des sains.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Crosstab sexe x target ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sexe × Maladie - `pd.crosstab`")
        ct = pd.crosstab(df_clean["sex"], df_clean["target"], normalize="index")
        ct.index   = ["Homme (0)", "Femme (1)"]
        ct.columns = ["Sain", "Malade"]
        st.dataframe(ct.round(3), use_container_width=True)

        fig, ax = plt.subplots(figsize=(5, 3))
        ct.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="white")
        ax.set_xticklabels(ct.index, rotation=0)
        ax.set_title("Proportion malades/sains par sexe")
        ax.set_ylabel("Proportion")
        ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("""
        <div style="background:#f8f9fa;border-left:4px solid #5c8ee0;border-radius:8px;
                    padding:12px 16px;margin-top:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
        📌 <b>Interprétation</b><br><br>
        <b>56 % des hommes</b> sont malades, contre <b>52 % des femmes</b>.
        Les hommes sont légèrement plus touchés, mais la différence reste faible.
        Le sexe seul ne suffit pas à prédire la maladie.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.subheader("Type douleur (cp) × Maladie - `pd.crosstab`")
        ct2 = pd.crosstab(df_clean["cp"], df_clean["target"], normalize="index")
        ct2.columns = ["Sain", "Malade"]
        st.dataframe(ct2.round(3), use_container_width=True)

        fig, ax = plt.subplots(figsize=(5, 3))
        ct2.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="white")
        ax.set_xticklabels([f"cp={i}" for i in ct2.index], rotation=0)
        ax.set_title("Proportion malades/sains par type de douleur")
        ax.set_ylabel("Proportion")
        ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("""
        <div style="background:#f8f9fa;border-left:4px solid #5c8ee0;border-radius:8px;
                    padding:12px 16px;margin-top:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
        📌 <b>Interprétation</b><br><br>
        Les patients avec <b>cp=0 (aucune douleur)</b> sont paradoxalement les plus malades (~72 %).
        Ceux avec <b>cp=3</b> sont majoritairement sains (~60 %).
        C'est contre-intuitif : l'absence de douleur peut masquer une maladie silencieuse.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── pd.cut - catégories d'âge ──
    st.subheader("Tranche d'âge × Maladie - `pd.cut`")
    st.code("""df['age_categ'] = pd.cut(
    df['age'],
    bins=[0, 40, 60, 100],
    labels=['jeune', 'adulte', 'senior']
)""")

    ct3 = pd.crosstab(df_clean["age_categ"], df_clean["target"], normalize="index")
    ct3.columns = ["Sain", "Malade"]

    col_a, col_b = st.columns(2)
    with col_a:
        st.dataframe(ct3.round(3), use_container_width=True)
        st.dataframe(
            df_clean["age_categ"].value_counts().rename("Nb patients").to_frame(),
            use_container_width=True
        )
    with col_b:
        fig, ax = plt.subplots(figsize=(5, 4))
        ct3.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="white")
        ax.set_xticklabels(ct3.index, rotation=0)
        ax.set_title("Proportion malades/sains par tranche d'âge")
        ax.set_ylabel("Proportion")
        ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("""
    <div style="background:#f8f9fa;border-left:4px solid #e05c5c;border-radius:8px;
                padding:14px 18px;margin-top:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
    📌 <b>Interprétation</b><br><br>
    Le risque de maladie cardiaque augmente clairement avec l'âge :<br>
    - <b>Jeunes (- de 40 ans)</b> : environ 40 % sont malades - c'est le groupe le moins touché.<br>
    - <b>Adultes (40-60 ans)</b> : environ 53 % sont malades - le risque dépasse déjà la moitié.<br>
    - <b>Seniors (+ de 60 ans)</b> : environ 60 % sont malades - c'est le groupe le plus à risque.<br><br>
    Plus on avance en âge, plus la probabilité d'avoir une maladie cardiaque augmente.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Moyenne par groupe personnalisé ──
    st.subheader("Exploration personnalisée - `groupby`")
    col_group = st.selectbox("Grouper par", ["target", "sex", "cp", "age_categ", "fbs"])
    col_stat  = st.selectbox("Statistique sur", ["age", "thalach", "chol", "oldpeak", "trestbps"])

    result = df_clean.groupby(col_group)[col_stat].agg(["mean", "median", "min", "max"]).round(2)
    st.dataframe(result, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 3))
    result["mean"].plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Moyenne de {col_stat} par {col_group}")
    ax.set_ylabel(f"Moyenne {col_stat}")
    ax.axhline(df_clean[col_stat].mean(), color="tomato", linestyle="--", label="Moyenne globale")
    ax.legend(); plt.xticks(rotation=0); plt.tight_layout()
    st.pyplot(fig); plt.close()

    moy_globale = df_clean[col_stat].mean()
    valeurs = result["mean"]
    groupe_max = valeurs.idxmax()
    groupe_min = valeurs.idxmin()
    ecart = valeurs.max() - valeurs.min()

    st.markdown(f"""
    <div style="background:#f8f9fa;border-left:4px solid #5c8ee0;border-radius:8px;
                padding:14px 18px;margin-top:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
    📌 <b>Interprétation</b><br><br>
    La ligne rouge pointillée représente la <b>moyenne globale de {col_stat} ({moy_globale:.1f})</b> pour tous les patients.<br><br>
    - Le groupe <b>{groupe_max}</b> a la moyenne la plus haute ({valeurs.max():.1f}) - au-dessus de la moyenne globale.<br>
    - Le groupe <b>{groupe_min}</b> a la moyenne la plus basse ({valeurs.min():.1f}) - en dessous de la moyenne globale.<br><br>
    L'écart entre les groupes est de <b>{ecart:.1f}</b>.
    {"Cet écart est significatif : la variable " + col_group + " influence clairement " + col_stat + "." if ecart / moy_globale > 0.05 else "Cet écart est faible : la variable " + col_group + " n'influence pas beaucoup " + col_stat + "."}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 - MODÈLE PRÉDICTIF
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Modèle prédictif":
    st.title("🤖 Modèle prédictif - KNN")

    st.markdown("""
    <div style="background:#eef4ff;border-left:4px solid #5c8ee0;border-radius:8px;
                padding:16px 20px;margin-bottom:16px;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
    <b>💡 Comment fonctionne le KNN ?</b><br><br>
    Le KNN (<i>K plus proches voisins</i>) est un algorithme simple : pour prédire si un nouveau patient est malade ou non,
    il regarde les <b>K patients les plus similaires</b> dans les données d'entraînement et choisit la réponse majoritaire.<br><br>
    Par exemple avec <b>k=5</b> : si parmi les 5 patients les plus proches, 4 sont malades, le modèle prédit "malade".<br><br>
    - <b>k petit</b> (ex: k=1) : le modèle est très sensible, il peut se tromper sur des cas isolés.<br>
    - <b>k grand</b> (ex: k=15) : le modèle est plus stable mais peut rater des cas particuliers.<br>
    - Le modèle est d'abord <b>entraîné</b> sur une partie des données, puis <b>évalué</b> sur les données qu'il n'a jamais vues.
    </div>
    """, unsafe_allow_html=True)

    X     = df_clean.drop(columns=["target", "age_categ"])
    y     = df_clean["target"]
    X_norm = 2 * (X - X.min()) / (X.max() - X.min()) - 1

    # Paramètres sidebar
    st.sidebar.markdown("### ⚙️ Paramètres du modèle")
    k         = st.sidebar.slider("Nombre de voisins (k)", 1, 15, 5)
    test_size = st.sidebar.slider("Taille du jeu de test (%)", 15, 40, 20) / 100

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=test_size, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # ── Métriques ──
    acc = accuracy_score(y_test, y_pred)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",       f"{acc:.1%}")
    c2.metric("Train",          len(X_train))
    c3.metric("Test",           len(X_test))
    c4.metric("k sélectionné",  k)

    st.markdown(f"""
    <div style="background:#f8f9fa;border-left:4px solid #e05c5c;border-radius:8px;
                padding:14px 18px;margin-top:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
    📌 <b>Interprétation</b><br><br>
    Le modèle a été entraîné sur <b>{len(X_train)} patients</b> et testé sur <b>{len(X_test)} patients</b> qu'il n'avait jamais vus.<br>
    Il a obtenu une accuracy de <b>{acc:.1%}</b>, ce qui signifie qu'il a correctement prédit le statut
    (malade ou sain) de <b>{int(acc * len(X_test))} patients sur {len(X_test)}</b>.
    {"C'est un bon résultat pour un modèle simple comme le KNN." if acc >= 0.75 else "Le résultat est correct mais peut être amélioré en changeant k ou en utilisant plus de données."}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # ── Matrice de confusion ──
    with col_left:
        st.subheader("Matrice de confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        labels = ["Sain", "Malade"]
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Classe prédite"); ax.set_ylabel("Vraie classe")
        ax.set_title(f"KNN (k={k})")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown(f"""
        <div style="background:#f8f9fa;border-left:4px solid #5c8ee0;border-radius:8px;
                    padding:12px 16px;margin-top:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
        📌 <b>Comment lire ce tableau ?</b><br><br>
        - <b>{cm[0,0]} sains</b> correctement identifiés comme sains ✅<br>
        - <b>{cm[1,1]} malades</b> correctement identifiés comme malades ✅<br>
        - <b>{cm[0,1]} sains</b> mal classés comme malades ⚠️ (faux positifs)<br>
        - <b>{cm[1,0]} malades</b> mal classés comme sains ❌ (faux négatifs - les plus dangereux)<br><br>
        En médecine, rater un malade (faux négatif) est plus grave que signaler un sain par erreur.
        </div>
        """, unsafe_allow_html=True)

    # ── Rapport ──
    with col_right:
        st.subheader("Rapport de classification")
        report = classification_report(
            y_test, y_pred,
            target_names=["Sain (0)", "Malade (1)"],
            zero_division=0, output_dict=True
        )
        rep_df = pd.DataFrame(report).T.round(2)
        st.dataframe(rep_df, use_container_width=True)
        prec_m = rep_df.loc["Malade (1)", "precision"]
        rec_m  = rep_df.loc["Malade (1)", "recall"]
        st.markdown(f"""
        <div style="background:#f8f9fa;border-left:4px solid #5c8ee0;border-radius:8px;
                    padding:12px 16px;margin-top:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
        📌 <b>Comment lire ce rapport ?</b><br><br>
        - <b>Precision ({prec_m:.0%} pour les malades)</b> : quand le modèle dit "malade", il a raison {prec_m:.0%} du temps.<br>
        - <b>Recall ({rec_m:.0%} pour les malades)</b> : parmi tous les vrais malades, le modèle en détecte {rec_m:.0%}.<br>
        - <b>F1-score</b> : moyenne entre precision et recall - plus c'est proche de 1, mieux c'est.<br><br>
        Le modèle détecte mieux les malades que les sains, ce qui est souhaitable dans un contexte médical.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Courbe accuracy vs k ──
    st.subheader("Influence de k sur l'accuracy")
    k_range = range(1, 21)
    scores  = []
    for ki in k_range:
        m = KNeighborsClassifier(n_neighbors=ki)
        m.fit(X_train, y_train)
        scores.append(accuracy_score(y_test, m.predict(X_test)))

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(list(k_range), scores, marker="o", color="steelblue", linewidth=2)
    ax.axvline(k, color="tomato", linestyle="--", linewidth=1.5, label=f"k={k} sélectionné")
    ax.set_xlabel("k (nombre de voisins)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy en fonction de k")
    ax.set_xticks(list(k_range))
    ax.legend(); plt.tight_layout()
    st.pyplot(fig); plt.close()

    best_k = int(np.argmax(scores)) + 1
    best_score = max(scores)
    st.markdown(f"""
    <div style="background:#f8f9fa;border-left:4px solid #e05c5c;border-radius:8px;
                padding:14px 18px;margin-top:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
    📌 <b>Interprétation</b><br><br>
    Chaque point sur la courbe représente l'accuracy du modèle pour une valeur de k différente.<br><br>
    - La meilleure accuracy obtenue est <b>{best_score:.1%}</b> avec <b>k={best_k}</b>.<br>
    - Avec k=1, le modèle "mémorise" les données et peut être trop précis sur l'entraînement mais moins bon sur de nouveaux patients.<br>
    - En augmentant k, le modèle devient plus stable mais peut perdre en précision.<br><br>
    L'idéal est de choisir le k qui donne la meilleure accuracy sans être trop petit ni trop grand.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 - SIMULATEUR PATIENT
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Simulateur patient":
    st.title("🔮 Simulateur patient")
    st.markdown("Renseignez les informations du patient pour obtenir une prédiction en temps réel.")

    st.markdown("""
    <div style="background:#eef4ff;border-left:4px solid #5c8ee0;border-radius:8px;
                padding:16px 20px;margin:12px 0;box-shadow:2px 2px 8px rgba(0,0,0,0.08);">
    <b>📋 Comment remplir ce formulaire ?</b><br><br>
    Renseignez les données médicales du patient dans les champs ci-dessous. Voici les éléments
    qui ont le <b>plus d'influence</b> sur la prédiction :<br><br>
    - <b>Fréquence cardiaque max (thalach)</b> : une valeur basse (en dessous de 140 bpm) est un signal d'alerte.<br>
    - <b>Dépression ST (oldpeak)</b> : plus cette valeur est élevée (au-dessus de 1,5), plus le risque augmente.<br>
    - <b>Vaisseaux colorés (ca)</b> : plus il y en a (1, 2 ou 3), plus les artères sont obstruées.<br>
    - <b>Type de douleur (cp)</b> : cp=0 (aucune douleur) est paradoxalement associé à plus de risque.<br>
    - <b>Angine à l'effort (exang)</b> : si le patient ressent une douleur dans la poitrine à l'effort (1 = oui), c'est un signe fort.<br><br>
    ⚠️ <i>Ces données sont idéalement issues d'un bilan médical (ECG, scintigraphie, prise de sang).</i>
    </div>
    """, unsafe_allow_html=True)

    # Entraînement sur toutes les données
    X_all  = df_clean.drop(columns=["target", "age_categ"])
    y_all  = df_clean["target"]
    X_norm = 2 * (X_all - X_all.min()) / (X_all.max() - X_all.min()) - 1
    knn    = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_norm, y_all)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Informations personnelles")
        age      = st.slider("Âge",                              18, 77,  54)
        sex      = st.selectbox("Sexe",                          ["Homme (0)", "Femme (1)"])
        cp       = st.selectbox("Type douleur thoracique (cp)",  [0, 1, 2, 3],
                                help="0=asymptomatique, 1=angine typique, 2=atypique, 3=autre")
        trestbps = st.slider("Pression au repos (mmHg)",         90, 200, 130)
        chol     = st.slider("Cholestérol (mg/dL)",              140, 420, 246)
        fbs      = st.selectbox("Glycémie à jeun > 120 (fbs)",   [0, 1])
        restecg  = st.selectbox("ECG au repos (restecg)",        [0, 1, 2])

    with col2:
        st.markdown("#### Résultats d'effort")
        thalach  = st.slider("Fréquence cardiaque max (bpm)",    80, 210, 150)
        exang    = st.selectbox("Angine à l'effort (exang)",     [0, 1])
        oldpeak  = st.slider("Dépression ST (oldpeak)",          0.0, 6.2, 1.0, step=0.1)
        slope    = st.selectbox("Pente ST (slope)",              [0, 1, 2])
        ca       = st.selectbox("Nb vaisseaux colorés (ca)",     [0, 1, 2, 3])
        thal     = st.selectbox("Thalassémie (thal)",            [1, 2, 3])

    sex_val = 0 if sex.startswith("Homme") else 1

    patient = pd.DataFrame(
        [[age, sex_val, cp, trestbps, chol, fbs, restecg,
          thalach, exang, oldpeak, slope, ca, thal]],
        columns=X_all.columns
    )

    # Normalisation avec les min/max du dataset d'entraînement
    patient_norm = 2 * (patient - X_all.min()) / (X_all.max() - X_all.min()) - 1
    patient_norm = patient_norm.clip(-1, 1)

    st.markdown("---")

    if st.button("🔍 Lancer la prédiction", use_container_width=True):
        prediction = knn.predict(patient_norm)[0]
        proba      = knn.predict_proba(patient_norm)[0]

        # ── Résultat ──
        if prediction == 1:
            st.error(f"### ❤️ Risque de maladie cardiaque détecté  |  Probabilité : **{proba[1]:.0%}**")
        else:
            st.success(f"### 💚 Pas de maladie cardiaque détectée  |  Probabilité : **{proba[0]:.0%}**")

        col_a, col_b = st.columns(2)

        # ── Graphique probabilités ──
        with col_a:
            fig, ax = plt.subplots(figsize=(4, 3))
            bars = ax.bar(["Sain", "Malade"], proba,
                           color=["steelblue", "tomato"], edgecolor="white")
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Probabilité")
            ax.set_title("Probabilités prédites")
            for bar, val in zip(bars, proba):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.03, f"{val:.0%}",
                        ha="center", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # ── Données saisies ──
        with col_b:
            st.markdown("**Données saisies :**")
            st.dataframe(
                patient.T.rename(columns={0: "Valeur"}),
                use_container_width=True
            )

    st.caption("⚠️ Modèle entraîné sur données synthétiques - usage éducatif uniquement.")
