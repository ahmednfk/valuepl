import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ===========================
# 1. Chargement des fichiers .pkl
# ===========================
ASSETS = {
    'best_model': 'best_model.pkl',
    'scaler': 'scaler.pkl',
    'imputer_num': 'imputer_num.pkl',
    'imputer_cat': 'imputer_cat.pkl',
    'label_encoders': 'label_encoders.pkl',
    'x_columns': 'x_columns.pkl',
    'numeric_features': 'numeric_features.pkl',
    'categorical_features': 'categorical_features.pkl'
}

for name, file in ASSETS.items():
    if not os.path.exists(file):
        st.error(f"Fichier '{file}' manquant ! Exécute d'abord generate_assets.py")
        st.stop()

best_model = joblib.load(ASSETS['best_model'])
scaler = joblib.load(ASSETS['scaler'])
imputer_num = joblib.load(ASSETS['imputer_num'])
imputer_cat = joblib.load(ASSETS['imputer_cat'])
label_encoders = joblib.load(ASSETS['label_encoders'])
X_COLUMNS = joblib.load(ASSETS['x_columns'])
numeric_features = joblib.load(ASSETS['numeric_features'])
categorical_features = joblib.load(ASSETS['categorical_features'])

MODEL_NAME = type(best_model).__name__

# ===========================
# 2. Fonction de préparation et prédiction
# ===========================
def prepare_and_predict(input_data: dict):
    # Créer un DataFrame avec toutes les colonnes
    data_full = {col: np.nan for col in X_COLUMNS}
    data_full.update(input_data)
    df = pd.DataFrame([data_full], columns=X_COLUMNS)
    
    # Imputation
    if numeric_features:
        df[numeric_features] = imputer_num.transform(df[numeric_features])
    if categorical_features:
        df[categorical_features] = imputer_cat.transform(df[categorical_features])
    
    # Encodage des catégorielles
    for col in categorical_features:
        le = label_encoders[col]
        val = df.at[0, col]
        if val not in le.classes_:
            val = imputer_cat.statistics_[categorical_features.index(col)]
        df[col] = le.transform([val])
    
    # Scaling
    df_scaled = scaler.transform(df)
    
    # Prédiction
    pred = best_model.predict(df_scaled)[0]
    return pred

# ===========================
# 3. Interface Streamlit
# ===========================
st.set_page_config(page_title="Prédiction Valeur Joueur", layout="wide")
st.title("⚽ Prédiction de Valeur Marchande des Joueurs")
st.markdown(f"**Modèle utilisé :** `{MODEL_NAME}`")
st.markdown("---")

with st.form("prediction_form"):
    st.subheader("Caractéristiques du joueur")
    col1, col2, col3, col4 = st.columns(4)
    
    user_pos = col1.selectbox("Position (Pos)", options=['Attaquant', 'Milieu', 'Défenseur', 'Gardien', 'Non renseigné'])
    user_squad = col2.selectbox("Équipe (Squad)", options=['FC Elite', 'AC Red', 'Blue United', 'Autre'])
    user_nation = col3.selectbox("Nationalité (Nation)", options=['France', 'Espagne', 'Allemagne', 'Autre'])
    user_comp = col4.selectbox("Compétition (Comp)", options=['Ligue 1', 'La Liga', 'Premier League', 'Autre'])
    
    st.subheader("Statistiques numériques")
    coln1, coln2, coln3, coln4 = st.columns(4)
    
    age = coln1.number_input("Âge", min_value=15, max_value=45, value=25)
    mp = coln2.number_input("Matchs joués (MP)", min_value=0, value=30)
    minutes = coln3.number_input("Minutes (Min)", min_value=0, value=2500)
    goals = coln4.number_input("Buts (Goals)", min_value=0, value=10)
    
    coln5, coln6, coln7, coln8 = st.columns(4)
    assists = coln5.number_input("Assists", min_value=0, value=5)
    tkl = coln6.number_input("Tkl", min_value=0, value=20)
    tkl_won = coln7.number_input("TklWon", min_value=0, value=15)
    aer_won = coln8.number_input("AerWon", min_value=0, value=30)
    
    aer_lost = st.number_input("AerLost", min_value=0, value=25)
    
    submitted = st.form_submit_button("Calculer Valeur")

if submitted:
    user_input = {
        'Pos': user_pos,
        'Squad': user_squad,
        'Nation': user_nation,
        'Comp': user_comp,
        'Age': age,
        'MP': mp,
        'Min': minutes,
        'Goals': goals,
        'Assists': assists,
        'Tkl': tkl,
        'TklWon': tkl_won,
        'AerWon': aer_won,
        'AerLost': aer_lost
    }
    
    try:
        prediction = prepare_and_predict(user_input)
        st.success(f"✅ Valeur marchande prédite : €{prediction:,.2f}")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
