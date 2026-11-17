"""
AgriYield-ML — Application Streamlit de prédiction du rendement agricole.
Lancer en local :  streamlit run app.py
"""
import os

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
st.set_page_config(page_title="AgriYield-ML", page_icon="🌾", layout="centered")

MODEL_PATH = "models/model_compressed.joblib"
DATA_PATH = "data/processed/agricultural_clean.csv"


# ---------------------------------------------------------------
# Chargement (mis en cache : une seule fois par session serveur)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


if not os.path.exists(MODEL_PATH):
    st.error(f"Modèle introuvable : {MODEL_PATH}")
    st.stop()

model = load_model()
df = load_data()

# Liste des variables attendues par le modèle (scikit-learn >= 1.0)
features = list(getattr(model, "feature_names_in_", []))
if not features:
    st.error("Le modèle ne contient pas la liste de ses variables (feature_names_in_).")
    st.stop()

# ---------------------------------------------------------------
# Interface
# ---------------------------------------------------------------
st.title("🌾 AgriYield-ML")
st.write("Prédiction du rendement agricole à partir des caractéristiques de la parcelle.")

st.sidebar.header("Paramètres d'entrée")
inputs = {}

def is_year_column(name, serie):
    """Colonne année : uniquement des entiers compris entre 1900 et 2100.
    (Le nom seul ne suffit pas : ex. 'Annual_Rainfall' n'est pas une année.)"""
    if not (pd.api.types.is_numeric_dtype(serie) and len(serie) > 0):
        return False
    entiers = (serie % 1 == 0).all()
    return bool(entiers and serie.min() >= 1900 and serie.max() <= 2100)


def is_integer_column(serie):
    return pd.api.types.is_numeric_dtype(serie) and len(serie) > 0 and (serie % 1 == 0).all()


for col in features:
    if df is not None and col in df.columns:
        serie = df[col].dropna()
        if is_year_column(col, serie):
            # Année : on autorise les années au-delà des données
            inputs[col] = st.sidebar.number_input(
                col,
                min_value=int(serie.min()),
                max_value=2050,
                value=int(serie.max()),
                step=1,
            )
            if inputs[col] > serie.max():
                st.sidebar.warning(
                    f"Les données s'arrêtent en {int(serie.max())} : "
                    "la prédiction pour une année postérieure est une extrapolation."
                )
        elif is_integer_column(serie):
            # Variable entière (nombre de jours, comptages...) : pas de décimales
            inputs[col] = st.sidebar.number_input(
                col,
                value=int(serie.median()),
                step=1,
            )
        elif pd.api.types.is_numeric_dtype(serie):
            inputs[col] = st.sidebar.number_input(
                col,
                value=float(serie.median()),
            )
        else:
            inputs[col] = st.sidebar.selectbox(col, sorted(serie.unique()))
    else:
        # Variable absente du CSV (ex. colonne encodée) : saisie libre
        inputs[col] = st.sidebar.number_input(col, value=0.0)

X = pd.DataFrame([inputs])[features]

st.subheader("Données saisies")
st.dataframe(X, use_container_width=True)

if st.button("Prédire le rendement", type="primary"):
    try:
        prediction = model.predict(X)[0]
        st.success(f"Rendement prédit : **{prediction:,.2f}**")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

# ---------------------------------------------------------------
# Aperçu des données
# ---------------------------------------------------------------
if df is not None:
    with st.expander("Voir les données d'entraînement"):
        st.dataframe(df.head(100), use_container_width=True)