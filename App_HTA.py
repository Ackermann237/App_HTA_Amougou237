import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
import base64
import joblib
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import os
from sklearn.preprocessing import StandardScaler
from pycaret.classification import predict_model, load_model  # Ajout de l'importation
from transformers import pipeline
import tensorflow as tf
from PIL import Image
import io
import requests
from gtts import gTTS  # Utilisation de gTTS
from dotenv import load_dotenv  # Import unique
from groq import Groq  # Ajout de l'importation Groq
from streamlit_lottie import st_lottie
import json
from datetime import datetime  # Importation explicite de la classe datetime
import traceback  # Ajout pour gérer les erreurs

# Charger les variables d'environnement
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Clé API Groq non trouvée. Vérifiez votre fichier .env avec GROQ_API_KEY.")
    st.stop()
elif len(api_key) < 20:
    st.error("La clé API Groq semble invalide. Vérifiez ou régénérez-la.")
    st.stop()

# Initialisation du client Groq
try:
    groq_client = Groq(api_key=api_key)  # Utilisation explicite de groq_client
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Groq : {e}")
    st.stop()

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de l'Hypertension Artérielle - GMSH",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS inspiré du tableau de bord de l'image, avec couleurs GMSH
st.markdown("""
<style>
    .main-container {
        padding: 1rem;
        background-color: #F5F6F5;
        font-family: 'Helvetica Neue', sans-serif;
        color: #34495E;
    }
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        font-weight: bold;
        color: #4e4e4f !important;
    }
    .sub-title {
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        font-style: italic;
        font-family: 'Helvetica Neue', sans-serif;
        margin-top: -10px;
        margin-bottom: 30px;
        padding: 10px;
        color: #060e80 !important;
    }
    .professional-layout {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }
    .professional-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .professional-metric {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 6px solid #060e80;
        text-align: center;
        width: 100%;
    }
    .professional-metric:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
    }
    .enhanced-metric {
        padding: 3.5rem;
        background-color: #F9FAFB;
        border-left: 8px solid #060e80;
    }
    .section-header {
        position: relative;
        color: #060e80 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: bold;
        font-size: 3em;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        text-transform: none;
    }
    .intro-text.professional-card h3 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .intro-text.professional-card p {
        font-size: 1.1rem;
        line-height: 1.8;
    }
    .header-image-container {
        position: relative;
        text-align: center;
        margin-bottom: 2rem;
    }
    .header-image-container img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
    }
    .image-overlay-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.5rem;
        color: #FFFFFF;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        background-color: rgba(6, 14, 128, 0.7);
        padding: 1rem 2rem;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #060e80;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
    }
    .sidebar .sidebar-content {
        padding: 1rem;
        background-color: #060e80;
        color: #FFFFFF;
        display: flex;
        flex-direction: column;
        align-items: center;
        height: 100%;
    }
    .sidebar .sidebar-content h1, .sidebar .sidebar-content h2, .sidebar .sidebar-content h3, .sidebar .sidebar-content h4, .sidebar .sidebar-content h5, .sidebar .sidebar-content h6 {
        color: #FFFFFF;
    }
    .sidebar .sidebar-content .stSelectbox label, .sidebar .sidebar-content .stNumberInput label {
        color: #FFFFFF;
    }
    .sidebar .sidebar-content .stSelectbox select, .sidebar .sidebar-content .stNumberInput input {
        background-color: #6c6969;
        color: #060e80;
        border: 1px solid #6c6969;
    }
    .sidebar .logo-container {
        width: 100%;
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
    }
    .sidebar .logo-container img {
        max-width: 50px;
        height: auto;
        margin: 0 auto;
    }
    .metric-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
        border-left: 5px solid #060e80;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-card h3 {
        color: #060e80;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        color: #34495E;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .footer {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1rem;
        background-color: #060e80;
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: #FFFFFF;
        border-top: 2px solid #6c6969;
        padding: 10px;
        z-index: 1000;
    }
    .animated {
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .image-container {
        position: relative;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Cache pour le chargement des données
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("./Donnée_Hypertention_IQR_mixed.xlsx")
        # Vérification et remplacement des valeurs manquantes ou invalides
        numeric_cols = ['Age', 'Taille', 'masse_corporelle', 'IMC', 'FR', 'SP02', 'Systolic', 'Diastolic']
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())  # Remplace NaN par la moyenne
        return df
    except FileNotFoundError:
        st.error("Le fichier 'Donnée_Hypertention_IQR_mixed.xlsx' est introuvable. Veuillez vérifier le chemin.")
        st.stop()

# Fonction pour charger une animation Lottie depuis un fichier local ou URL
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_heart = load_lottiefile("./Animation - 1751629587652.json")

# Cache pour le chargement des modèles
@st.cache_resource(ttl=3600)  # Cache avec un TTL de 1 heure
def load_models(_cache_version=1):  # Ajoute un paramètre de version
    required_files = ['scaler.pkl', 'model_HTA_Pycaret_final.pkl', 'rf_smote.pkl', 'svm_model.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"Les fichiers suivants sont manquants : {', '.join(missing_files)}. Veuillez les vérifier.")
        return None, None, None, None
    
    try:
        scaler = joblib.load('scaler.pkl')
        pycaret_model = joblib.load('model_HTA_Pycaret_final.pkl')
        rf_model = joblib.load('rf_smote.pkl')
        svm_model = joblib.load('svm_model.pkl')
      
        return scaler, pycaret_model, rf_model, svm_model
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles ou du scaler : {str(e)}")
        return None, None, None, None

# Génération du lien de téléchargement CSV
def file_download(df, filename="donnees_hta.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger le fichier CSV</a>'

# Définition de la fonction gtts_tts pour convertir le texte en audio
def gtts_tts(text):
    try:
        tts = gTTS(text=text, lang='fr')  # Langue française par défaut
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        return audio_file
    except Exception as e:
        st.error(f"Erreur avec gTTS : {e}")
        return None

# Fonction pour prédire avec hyperparamètres
def predict_with_model(model, model_name, input_data, hyperparameters):
    if model_name == "PyCaret":
        prediction = model.predict(input_data)
    elif model_name == "Random Forest (SMOTE)":
        if "n_estimators" in hyperparameters:
            model.n_estimators = hyperparameters["n_estimators"]
        prediction = model.predict(input_data)
    elif model_name == "SVM":
        if "C" in hyperparameters:
            model.C = hyperparameters["C"]
        prediction = model.predict(input_data)
    return prediction

def main():
    import pandas as pd
    import seaborn as sns
    import numpy as np
    # Chargement des données et modèles
    df = load_data()
    pycaret_model, scaler, rf_model, svm_model = load_models(_cache_version=2)
    if pycaret_model is None or rf_model is None or svm_model is None:
        st.stop()

    # Prétraitement des données pour les visualisations
    # Calcul de poid_Categorie si absent
    df['poid_Categorie'] = df['IMC'].apply(lambda x: 'Obésité' if x >= 30 else 'Surpoids' if 25 <= x < 30 else 'Normal' if 18.5 <= x < 25 else 'Sous-poids')
    df = pd.get_dummies(df, columns=['poid_Categorie'])

    # Configuration de la barre latérale
    with st.sidebar:
        # Conteneur pour le logo centré
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image('./logo_gmsh-removebg-preview.png', width=150, use_container_width=True)
        st.markdown("<h2 style='color:#060e80; text-align: center;'>🩺 Groupe Médical St-Hilaire</h2>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
        menu = ["🏠 Accueil", "📊 Visualisations", "🤖 Prédiction ML", "🧠 Prédiction Deep Learning", "🤖 Chat Médical"]
        choice = st.selectbox("Navigation", menu)

    # Conteneur principal
    with st.container():
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown('<h1 class="main-title">Bienvenue sur HyperSight AI : Application de Prédiction de l\'Hypertension Artérielle (HTA) 🩺</h1>', unsafe_allow_html=True)
        with col2:
            st_lottie(lottie_heart, speed=3, width=120, height=100, key="heart")

        st.markdown('<h2 class="sub-title">Analyse, Visualisation et Prédiction via l\'IA : Données Cliniques & Imagerie</h2>', unsafe_allow_html=True)

        if choice == "🏠 Accueil":
            # Image centrée et agrandie
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.image('./ImageHTA1.png', width=700)

            # Introduction
            st.markdown('<h3 class="section-header">C\'est quoi hypersight AI ?</h3>', unsafe_allow_html=True)
            st.markdown("""
            <div class="intro-text animated">
                HyperSight AI est une application innovante qui combine l'apprentissage automatique et l'imagerie rétinienne pour prédire le risque d'hypertension artérielle.
                Elle aide les professionnels de santé à prendre des décisions éclairées grâce à l'analyse intelligente de données cliniques et visuelles.
                Explorez les données, visualisez les indicateurs et optimisez la prise en charge grâce à l’intelligence artificielle.
            </div>
            """, unsafe_allow_html=True)

            # Importance de l'application
            st.markdown('<h3 class="section-header">Pourquoi prédire l\'HTA ?</h3>', unsafe_allow_html=True)
            st.markdown("""
                <div class="intro-text animated">
                    <p>L'hypertension artérielle est un facteur de risque majeur pour les maladies cardiovasculaires. Une détection précoce grâce à des prédictions précises permet de sauver des vies, d'optimiser les traitements et de réduire les coûts de santé. Cette application vise à transformer la prise en charge des patients à travers des données fiables et des insights actionnables.</p>
                </div>
                """, unsafe_allow_html=True)

            # Section Métriques avec deux rangées
            st.markdown('<h3 class="section-header">Aperçu des Indicateurs Clés</h3>', unsafe_allow_html=True)

            # Première rangée : 3 métriques de base
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                    <div class="metric-card professional-metric" style="padding: 2rem; font-size: 1.2rem;">
                        <h3 style="font-size: 1.6rem; color: #060e80;">🧑‍🤝‍🧑 Total Patients</h3>
                        <p style="font-size: 2.5rem; color: #34495E;">{}</p>
                    </div>
                """.format(len(df)), unsafe_allow_html=True)
            with col2:
                st.markdown("""
                    <div class="metric-card professional-metric" style="padding: 2rem; font-size: 1.2rem;">
                        <h3 style="font-size: 1.6rem; color: #060e80;">📈 Taux HTA</h3>
                        <p style="font-size: 2.5rem; color: #34495E;">{:.1f}%</p>
                    </div>
                """.format(df['Hypertension_Arterielle'].mean() * 100), unsafe_allow_html=True)
            with col3:
                st.markdown("""
                    <div class="metric-card professional-metric" style="padding: 2rem; font-size: 1.2rem;">
                        <h3 style="font-size: 1.6rem; color: #060e80;">💗 Pression Diastolique</h3>
                        <p style="font-size: 2.5rem; color: #34495E;">{:.1f} mmHg</p>
                    </div>
                """.format(df['Diastolic'].mean()), unsafe_allow_html=True)

            # Deuxième rangée : 3 métriques agrandies
            col4, col5, col6 = st.columns(3)
            with col4:
                st.markdown("""
                    <div class="metric-card professional-metric enhanced-metric" style="padding: 3.5rem; font-size: 1.5rem;">
                        <h3 style="font-size: 1.6rem; color: #060e80;">👴 Âge Moyen</h3>
                        <p style="font-size: 2.5rem; color: #34495E;">{:.1f} ans</p>
                    </div>
                """.format(df['Age'].mean()), unsafe_allow_html=True)
            with col5:
                st.markdown("""
                    <div class="metric-card professional-metric enhanced-metric" style="padding: 3.5rem; font-size: 1.5rem;">
                        <h3 style="font-size: 1.6rem; color: #060e80;">🩺 IMC Moyen</h3>
                        <p style="font-size: 2.5rem; color: #34495E;">{:.1f}</p>
                    </div>
                """.format(df['IMC'].mean()), unsafe_allow_html=True)
            with col6:
                st.markdown("""
                    <div class="metric-card professional-metric enhanced-metric" style="padding: 3.5rem; font-size: 1.5rem;">
                        <h3 style="font-size: 1.6rem; color: #060e80;">💓 Pression Systolique</h3>
                        <p style="font-size: 2.5rem; color: #34495E;">{:.1f} mmHg</p>
                    </div>
                """.format(df['Systolic'].mean()), unsafe_allow_html=True)

            # Section Visualisations avec disposition améliorée
            st.markdown('<h3 class="section-header">Répartition des Patients par Statut HTA</h3>', unsafe_allow_html=True)
            col1, col2 = st.columns([1.5, 1])
            with col1:
                hta_counts = df['Hypertension_Arterielle'].value_counts()
                fig = px.pie(values=hta_counts.values, names=["Non Hypertendu", "Hypertendu"],
                            color_discrete_sequence=["#060e80", "#6c6969"])
                st.plotly_chart(fig, use_container_width=True)
       
            # Section Aperçu des données
            st.markdown('<h3 class="section-header">Aperçu des Données</h3>', unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            st.markdown('<h4 style="color: #060e80; font-size: 1.2rem;">📥 Télécharger le Jeu de Données</h4>', unsafe_allow_html=True)
            st.markdown(file_download(df), unsafe_allow_html=True)

            # Section Localisation
            st.markdown('<h3 class="section-header">Localisation du GMSH</h3>', unsafe_allow_html=True)
            st.markdown("""
                <div class="professional-card">
                    <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3980.6052749002088!2d11.506115974656662!3d3.8944513481199072!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x108bcf440d4a6907%3A0xb86846bef2f79b27!2sGroupe%20M%C3%A9dical%20St%20Hilaire!5e0!3m2!1sfr!2scm!4v1749386990850!5m2!1sfr!2scm" width="100%" height="400" style="border:0; border-radius: 10px;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
                </div>
            """, unsafe_allow_html=True)

            # Informations supplémentaires (À propos)
            st.markdown('<h3 class="section-header">À propos</h3>', unsafe_allow_html=True)
            st.markdown("""
                <div class="metric-card">
                    <p style="color: #34495E;">Développée par <strong>Amougou André</strong> pour le <strong>Groupe Médical St-Hilaire</strong>.</p>
                    <p>📅 Date : 09/06/2025 | 📞 Contact : <a href="tel:+237659686322" style="color: #060e80;">+237 659686322</a></p>
                    <p>🔢 Version : 1.0 | 🌐  <a href="https://www.findhealthclinics.com/CM/Yaound%C3%A9/100629631693840/Groupe-M%C3%A9dical-St-Hilaire" style="color: #060e80;">Visitez notre site</a></p>
                </div>
            """, unsafe_allow_html=True)

        elif choice == "📊 Visualisations":
            import matplotlib.pyplot as plt
            st.markdown('<h3 class="section-header">Visualisations des Données</h3>', unsafe_allow_html=True)

            # Note pour le médecin sur l'utilisation
            st.markdown("""
                <div class="info-note professional-card">
                    <p style="color: #34495E; font-size: 1.1rem; line-height: 1.6;">
                        <strong>Guide pour les médecins :</strong> Cette section vous permet d'explorer des visualisations interactives des données sur l'hypertension artérielle. Utilisez le menu déroulant ci-dessous pour sélectionner la visualisation souhaitée. Commencez par le "Statut HTA - Résumé" pour une vue d'ensemble, puis choisissez une visualisation spécifique pour analyser les tendances (par exemple, répartition des patients ou corrélations). Consultez les légendes pour interpréter les données et prenez des décisions cliniques informées.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Créer la colonne HTA_Status et poid_Categorie globalement
            if 'HTA_Status' not in df.columns:
                df['HTA_Status'] = df['Hypertension_Arterielle'].map({0: 'Non Hypertendu', 1: 'Hypertendu'})
            if 'poid_Categorie' not in df.columns:
                df['poid_Categorie'] = df['IMC'].apply(lambda x: 'Obésité' if x >= 30 else 'Surpoids' if 25 <= x < 30 else 'Normal' if 18.5 <= x < 25 else 'Sous-poids')

            # Statut HTA - Résumé en premier (toujours affiché)
            st.markdown('<h4 style="color: #060e80;">📊 Statut HTA - Résumé</h4>', unsafe_allow_html=True)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown('<div class="metric-card" style="background-color: #060e80; color: white; text-align: center; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">Total Patients<br><span style="font-size: 1.8rem;">{}</span></div>'.format(len(df)), unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="metric-card" style="background-color: #060e80; color: white; text-align: center; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">Non Hypertendu<br><span style="font-size: 1.8rem;">{}</span></div>'.format(len(df) - df['Hypertension_Arterielle'].sum()), unsafe_allow_html=True)
            with col_c:
                st.markdown('<div class="metric-card" style="background-color: #060e80; color: white; text-align: center; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">Hypertendus<br><span style="font-size: 1.8rem;">{}</span></div>'.format(df['Hypertension_Arterielle'].sum()), unsafe_allow_html=True)

            # Selectbox pour choisir les visualisations avec style amélioré
            visualization_options = [
                "Distribution du Risque HTA",
                "Moyenne HTA par Groupe d'Âge",
                "Patients par Catégorie d'IMC",
                "Moyenne IMC par Poids et HTA",
                "Répartition Poids et HTA",
                "Tension Systolique par HTA",
                "Comparaison HTA",
                "IMC vs Risque HTA",
                "Âge par Statut HTA",
                "Caractéristique vs Risque HTA"
            ]
            selected_visualization = st.selectbox(
                "Sélectionnez une visualisation",
                visualization_options,
                format_func=lambda x: f"🔍 {x}",
                key="visualization_select",
                help="Choisissez une visualisation pour explorer les données d'hypertension."
            )

            # Style CSS pour agrandir le selectbox
            st.markdown("""
                <style>
                    div.stSelectbox > div[data-baseweb="select"] {
                        font-size: 1.2rem;
                        padding: 0.5rem 1rem;
                        border: 2px solid #060e80;
                        border-radius: 10px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    div.stSelectbox > div[data-baseweb="select"]:hover {
                        background-color: #f0f2f6;
                        border-color: #0056b3;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Affichage de la visualisation choisie
            col1, col2 = st.columns([2, 1])
            with col1:
                if selected_visualization == "Distribution du Risque HTA":
                    st.markdown('<h4 style="color: #060e80;">📊 Distribution du Risque HTA</h4>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.histplot(df['Hypertension_Arterielle'], kde=True, color="#060e80", ax=ax)
                    plt.title("Distribution du Risque d'Hypertension Artérielle", fontsize=14, pad=15)
                    plt.xlabel("Risque (0 = Non, 1 = Oui)", fontsize=12)
                    plt.ylabel("Fréquence", fontsize=12)
                    ax.set_ylim(0, 600)
                    ax.set_xlim(-0.1, 1.1)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                elif selected_visualization == "Moyenne HTA par Groupe d'Âge":
                    st.markdown('<h4 style="color: #060e80;">📊 Moyenne HTA par Groupe d\'Âge</h4>', unsafe_allow_html=True)
                    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=['Moins 30', '30-40', '41-50', 'Plus de 50'])
                    mean_htn = df.groupby('Age_Group')['Hypertension_Arterielle'].mean().reindex(['Moins 30', '30-40', '41-50', 'Plus de 50']).fillna(0)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(x=mean_htn.index, y=mean_htn.values, color='#060e80', ax=ax)
                    ax.set_xlabel("Groupe d'Âge", fontsize=12)
                    ax.set_ylabel("Moyenne HTA", fontsize=12)
                    plt.title("Moyenne HTA par Groupe d'Âge", fontsize=14, pad=15)
                    plt.xticks(rotation=45)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                elif selected_visualization == "Patients par Catégorie d'IMC":
                    st.markdown('<h4 style="color: #060e80;">📊 Patients par Catégorie d\'IMC</h4>', unsafe_allow_html=True)
                    imc_counts = df['poid_Categorie'].value_counts().reindex(['Normal', 'Surpoids', 'Obésité', 'Sous-poids'], fill_value=0)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(x=imc_counts.index, y=imc_counts.values, color='#060e80', ax=ax)
                    ax.set_xlabel("Catégorie d'IMC", fontsize=12)
                    ax.set_ylabel("Nombre de Patients", fontsize=12)
                    plt.title("Patients par Catégorie d'IMC", fontsize=14, pad=15)
                    plt.xticks(rotation=45)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                elif selected_visualization == "Moyenne IMC par Poids et HTA":
                    st.markdown('<h4 style="color: #060e80;">📊 Moyenne IMC par Poids et HTA</h4>', unsafe_allow_html=True)
                    weight_categories = ['Obésité', 'Surpoids', 'Normal', 'Sous-poids']
                    imc_means = df.groupby(['poid_Categorie', 'Hypertension_Arterielle'])['IMC'].mean().unstack(fill_value=0)
                    imc_non_htn = [imc_means.loc[cat, 0] if 0 in imc_means.columns else df[df['Hypertension_Arterielle'] == 0]['IMC'].mean() for cat in weight_categories]
                    imc_htn = [imc_means.loc[cat, 1] if 1 in imc_means.columns else df[df['Hypertension_Arterielle'] == 1]['IMC'].mean() for cat in weight_categories]
                    df_imc = pd.DataFrame({
                        'IMC': imc_non_htn + imc_htn,
                        'Catégorie': weight_categories * 2,
                        'HTA_Status': ['Non Hypertendu'] * 4 + ['Hypertendu'] * 4
                    })
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(x='Catégorie', y='IMC', hue='HTA_Status', data=df_imc, palette={'Non Hypertendu': '#060e80', 'Hypertendu': '#808080'}, ax=ax)
                    ax.set_xlabel("Catégorie de Poids", fontsize=12)
                    ax.set_ylabel("Moyenne IMC", fontsize=12)
                    plt.title("Moyenne IMC par Poids et HTA", fontsize=14, pad=15)
                    plt.xticks(rotation=45)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                elif selected_visualization == "Répartition Poids et HTA":
                    st.markdown('<h4 style="color: #060e80;">📊 Répartition Poids et HTA</h4>', unsafe_allow_html=True)
                    weight_counts = df.groupby(['poid_Categorie', 'Hypertension_Arterielle']).size().unstack(fill_value=0)
                    weight_categories = ['Obésité', 'Surpoids', 'Normal', 'Sous-poids']
                    non_htn_counts = [weight_counts.loc[cat, 0] if 0 in weight_counts.columns else 0 for cat in weight_categories]
                    htn_counts = [weight_counts.loc[cat, 1] if 1 in weight_counts.columns else 0 for cat in weight_categories]
                    bar_width = 0.35
                    x = np.arange(len(weight_categories))
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plt.bar(x - bar_width/2, non_htn_counts, bar_width, label='Non Hypertendu', color='#060e80')
                    plt.bar(x + bar_width/2, htn_counts, bar_width, label='Hypertendu', color='#808080')
                    for i, (htn, non_htn) in enumerate(zip(htn_counts, non_htn_counts)):
                        ax.text(i + bar_width/2, htn, str(htn), ha='center', va='bottom' if htn > 0 else 'top', color='white')
                        ax.text(i - bar_width/2, non_htn, str(non_htn), ha='center', va='bottom' if non_htn > 0 else 'top', color='white')
                    ax.set_xlabel('Catégorie de Poids', fontsize=12)
                    ax.set_ylabel('Nombre de Patients', fontsize=12)
                    ax.set_title('Répartition des Patients par Poids et HTA', fontsize=14, pad=15)
                    ax.set_xticks(x)
                    ax.set_xticklabels(weight_categories, rotation=0)
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                elif selected_visualization == "Tension Systolique par HTA":
                    st.markdown('<h4 style="color: #060e80;">📊 Tension Systolique par HTA</h4>', unsafe_allow_html=True)
                    systolic_stats = df.groupby('HTA_Status')[['Systolic']].agg(['median', 'mean', 'min', 'max']).reset_index()
                    systolic_stats.columns = ['HTA_Status', 'Median_Systolic', 'Moyenne_Systolic', 'Min_Systolic', 'Max_Systolic']
                    df = df.merge(systolic_stats, on='HTA_Status', how='left')
                    df_melted = pd.melt(df, id_vars=['HTA_Status'], value_vars=['Median_Systolic', 'Moyenne_Systolic', 'Min_Systolic', 'Max_Systolic'], 
                                        var_name='Statistic', value_name='Value')
                    fig, ax = plt.subplots(figsize=(8, 6))
                    palette = {'Median_Systolic': "#808080", 'Moyenne_Systolic': '#060e80', 'Min_Systolic': "#000000", 'Max_Systolic': "#571280"}
                    sns.barplot(x='HTA_Status', y='Value', hue='Statistic', data=df_melted, palette=palette, ax=ax)
                    ax.set_xlabel("Statut HTA", fontsize=12)
                    ax.set_ylabel("Valeur", fontsize=12)
                    plt.title("Tension Systolique par Statut HTA", fontsize=14, pad=15)
                    st.pyplot(fig)
                elif selected_visualization == "Comparaison HTA":
                    st.markdown('<h4 style="color: #060e80;">📊 Comparaison HTA</h4>', unsafe_allow_html=True)
                    labels = ['Non Hypertendu', 'Hypertendu']
                    sizes = [(len(df) - df['Hypertension_Arterielle'].sum()), df['Hypertension_Arterielle'].sum()]
                    colors = ['#060e80', '#808080']
                    fig, ax = plt.subplots(figsize=(5, 5))  # Réduit la taille du camembert
                    patches, texts, autotexts = ax.pie(sizes, labels=None, colors=colors, autopct=lambda pct: f'{pct:.1f}%', textprops={'color': 'white', 'fontsize': 10}, wedgeprops={'edgecolor': 'white'}, startangle=12)
                    ax.axis('equal')
                    plt.title("Comparaison des Patients par Statut HTA", fontsize=14, pad=15)
                    ax.legend(patches, labels, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, title="Statut HTA", frameon=True, edgecolor='white', labelcolor='black', fontsize=10)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                elif selected_visualization == "IMC vs Risque HTA":
                    st.markdown('<h4 style="color: #060e80;">📊 IMC vs Risque HTA</h4>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.scatterplot(x=df['IMC'], y=df['Hypertension_Arterielle'], hue=df['Hypertension_Arterielle'], palette={0: '#060e80', 1: '#808080'}, s=100, ax=ax)
                    plt.title("IMC vs Risque d'Hypertension Artérielle", fontsize=14, pad=15)
                    plt.xlabel("IMC", fontsize=12)
                    plt.ylabel("Risque HTA (0 = Non, 1 = Oui)", fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                elif selected_visualization == "Âge par Statut HTA":
                    st.markdown('<h4 style="color: #060e80;">📊 Âge par Statut HTA</h4>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.histplot(data=df, x="Age", hue="Hypertension_Arterielle", multiple="stack", palette=["#060e80", "#808080"], ax=ax)
                    plt.title("Distribution de l'Âge par Statut HTA", fontsize=14, pad=15)
                    plt.xlabel("Âge (années)", fontsize=12)
                    plt.ylabel("Nombre de Patients", fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                elif selected_visualization == "Caractéristique vs Risque HTA":
                    st.markdown('<h4 style="color: #060e80;">📊 Caractéristique vs Risque HTA</h4>', unsafe_allow_html=True)
                    feature = st.selectbox("Sélectionner une Caractéristique", ["Age", "Taille", "masse_corporelle", "IMC", "FR", "SP02", "Systolic", "Diastolic"])
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.scatterplot(x=df[feature], y=df["Hypertension_Arterielle"], hue=df["Hypertension_Arterielle"], palette={0: '#060e80', 1: '#808080'}, s=100, ax=ax)
                    plt.title(f"{feature} vs Risque d'Hypertension Artérielle", fontsize=14, pad=15)
                    plt.xlabel(feature, fontsize=12)
                    plt.ylabel("Risque HTA (0 = Non, 1 = Oui)", fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)

            st.markdown('<p style="color: #34495E; font-size: 1.1rem; line-height: 1.6;">Note : Ces visualisations interactives permettent d’explorer les relations entre les caractéristiques cliniques et le risque d’hypertension artérielle. Utilisez-les pour identifier des tendances et soutenir vos décisions médicales.</p>', unsafe_allow_html=True)
        elif choice == "🤖 Prédiction ML":
            from joblib import load
            import matplotlib.pyplot as plt
            import numpy as np
            st.warning(
                "⚠️ Cette application fournit des estimations générales basées sur vos données, mais elle ne constitue pas un diagnostic médical. "
                "Utilisez ces résultats uniquement à titre informatif et consultez un professionnel de santé pour toute décision. "
                "amougouAI décline toute responsabilité en cas d'usage inapproprié ou de conséquences."
            )

            st.markdown(
                '<h2 style="color: #060e80; font-weight: bold; border-bottom: 2px solid #060e80; padding-bottom: 5px;">Prédiction de l\'Hypertension Artérielle</h2>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<p style="color: #34495E;">Évaluez le risque d\'hypertension artérielle avec des données cliniques.</p>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<h4 style="color: #060e80; font-weight: bold; border-bottom: 1px solid #060e80; padding: 5px;">📋 Sélection du Modèle</h4>',
                unsafe_allow_html=True
            )
            model_options = ["Random Forest (SMOTE)"]
            selected_model = st.selectbox("Choisissez un modèle prédictif", model_options)

            if selected_model == "Random Forest (SMOTE)":
                hyper_n_estimators = st.slider("Nombre d'arbres (n_estimators)", 10, 200, 100)

            st.markdown(
                '<h4 style="color: #060e80; font-weight: bold; border-bottom: 2px solid #060e80; padding: 5px;">📋 Saisie des Données du Patient</h4>',
                unsafe_allow_html=True
            )

            with st.container():
                st.markdown('<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("👶 Âge (ans)", min_value=12, max_value=120, value=40, step=1, help="Âge du patient en années")
                    sexe = st.selectbox("👤 Sexe", ["M", "F"], index=0, help="Sexe du patient")
                    rx_enceinte = st.checkbox("🤰 Patient enceinte ?", help="Cochez si la patiente est enceinte")
                with col2:
                    masse_corporelle = st.number_input("⚖️ Masse Corporelle (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1, help="Poids du patient en kilogrammes")
                    taille = st.number_input("📏 Taille (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1, help="Taille du patient en centimètres")
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    imc = st.number_input("📊 IMC (kg/m²)", min_value=15.0, max_value=60.0, value=25.0, step=0.1, help="Indice de masse corporelle")
                    fr = st.number_input("🌬️ Fréquence Respiratoire (par min)", min_value=10, max_value=100, value=60, step=1, help="Respirations par minute")
                    sp02 = st.number_input("🩺 Saturation en O₂ (%)", min_value=70, max_value=100, value=95, step=1, help="Saturation en oxygène dans le sang")
                with col2:
                    systolic = st.number_input("💓 Pression Systolique (mmHg)", min_value=60, max_value=250, value=120, step=1, help="Pression artérielle systolique")
                    diastolic = st.number_input("💗 Pression Diastolique (mmHg)", min_value=30, max_value=150, value=80, step=1, help="Pression artérielle diastolique")
                    temperature = st.number_input("🌡️ Température (°C)", min_value=35.0, max_value=42.0, value=36.8, step=0.1, help="Température corporelle")
                st.markdown('</div>', unsafe_allow_html=True)

            if systolic < 60 or diastolic < 30:
                st.error("Erreur : Les valeurs de pression artérielle sont trop basses.")
                st.stop()

            raw_data = pd.DataFrame({
                'Age': [age], 'Masse Corporelle (kg)': [masse_corporelle], 'IMC': [imc], 'Pression Systolique (mmHg)': [systolic],
                'Pression Diastolique (mmHg)': [diastolic], 'Fréquence Respiratoire': [fr], 'SpO2 (%)': [sp02],
                'Température (°C)': [temperature], 'Taille (cm)': [taille], 'Sexe': [sexe], 'Enceinte': [rx_enceinte]
            })
            poid_categorie = 'Obésité' if imc >= 30 else 'Surpoids' if 25 <= imc < 30 else 'Normal' if 18.5 <= imc < 25 else 'Sous-poids'
            raw_data['Catégorie de Poids'] = poid_categorie

            input_data = pd.DataFrame({
                'Age': [age], 'masse_corporelle': [masse_corporelle], 'IMC': [imc], 'Systolic': [systolic],
                'Diastolic': [diastolic], 'FR': [fr], 'SP02': [sp02], 'temperature': [temperature],
                'Taille': [taille], 'SEXE': [sexe], 'RX_ENCEINTE': [1 if rx_enceinte else 0]
            })
            input_data['poid_Categorie'] = poid_categorie
            input_data = pd.get_dummies(input_data, columns=['poid_Categorie', 'SEXE'], drop_first=True)

            rx_columns = ['RX_ENCEINTE', 'RX_Normale', 'RX_REFUS', 'RX_PNEUMOPATHIE DROITE']
            for col in rx_columns:
                if col not in input_data.columns:
                    input_data[col] = 0

            expected_columns = ['Age', 'masse_corporelle', 'IMC', 'Systolic', 'Diastolic', 'FR', 'SP02', 'temperature',
                            'poid_Categorie_Sous-poids', 'poid_Categorie_Surpoids', 'poid_Categorie_Obésité',
                            'RX_ENCEINTE', 'RX_Normale', 'RX_PNEUMOPATHIE DROITE', 'RX_REFUS', 'SEXE_M', 'Taille']
            input_data = input_data.reindex(columns=expected_columns, fill_value=0)

            try:
                scaler = load('scaler.pkl')
            except Exception as e:
                st.error(f"Erreur lors du chargement du scaler : {str(e)}")
                st.stop()

            numeric_cols = ['Age', 'masse_corporelle', 'IMC', 'Systolic', 'Diastolic', 'FR', 'SP02', 'temperature', 'Taille']
            input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
            st.write("Données standardisées :\n", input_data[numeric_cols])

            if st.button("📊 Effectuer la Prédiction"):
                with st.spinner("Calcul en cours..."):
                    try:
                        rf_model = load('rf_smote.pkl')

                        if rf_model:
                            prediction = rf_model.predict_proba(input_data)[:, 1]
                        else:
                            st.error("Modèle non disponible ou non chargé correctement.")
                            st.stop()

                        probability = prediction[0]
                        st.success("Prédiction du modèle effectuée avec succès !")

                        st.markdown(
                            '<h4 style="color: #060e80; font-weight: bold; border-bottom: 1px solid #060e80; padding: 5px;">📋 Paramètres Saisis</h4>',
                            unsafe_allow_html=True
                        )
                        st.dataframe(raw_data)

                        risk_level = "Élevé" if probability >= 0.7 else "Modéré" if probability >= 0.3 else "Faible"
                        color, message = (
                            "#FF6B6B", f"Risque d'hypertension {risk_level.lower()} ({probability:.2f})"
                        ) if probability >= 0.7 else (
                            "#808080", f"Risque d'hypertension {risk_level.lower()} ({probability:.2f})"
                        ) if probability >= 0.3 else (
                            "#51CF66", f"Risque d'hypertension {risk_level.lower()} ({probability:.2f})"
                        )
                        st.markdown(f'<p style="color: {color}; font-size: 1.2rem; font-weight: bold;">{message}</p>', unsafe_allow_html=True)

                        if probability < 0.3:
                            st.info("Maintenez un mode de vie sain (alimentation équilibrée, exercice régulier).")
                        elif 0.3 <= probability < 0.7:
                            st.warning("Surveillez votre pression artérielle et consultez un médecin si nécessaire. Réduisez le sel.")
                        else:
                            st.error("Consultez un professionnel de santé rapidement pour un suivi. Réduisez le stress et le sel.")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(
                                f"""
                                <div class="prediction-card" style="background: linear-gradient(135deg, #e0e7ff 0%, #f0f2f6 100%); padding: 20px; border-radius: 15px; box-shadow: 0 6px 12px rgba(0,0,0,0.2); text-align: center; border: 3px solid #060e80;">
                                    <h3 style="color: #060e80; font-size: 2rem; margin: 0; font-weight: bold; text-transform: uppercase;">Risque Prédit</h3>
                                    <p style="color: #34495E; font-size: 3rem; font-weight: bold; margin: 15px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">{probability:.2f}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        with col2:
                            st.markdown(
                                f"""
                                <div class="prediction-card" style="background: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center; border: 2px solid #060e80;">
                                    <h3 style="color: #060e80; font-size: 1.5rem; margin: 0; font-weight: bold;">Prévalence Moyenne</h3>
                                    <p style="color: #34495E; font-size: 2rem; font-weight: bold; margin: 10px 0;">{df['Hypertension_Arterielle'].mean():.2f}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number", value=probability, title={"text": "Niveau de Risque d'Hypertension"},
                            gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#060e80"},
                                'steps': [{'range': [0, 0.3], 'color': "#51CF66"},
                                        {'range': [0.3, 0.7], 'color': "#808080"},
                                        {'range': [0.7, 1], 'color': "#FF6B6B"}],
                                'threshold': {'line': {'color': "#FF6B6B", 'width': 2}, 'value': 0.7}}
                        ))
                        st.plotly_chart(fig)

                        # Ajout du bouton de téléchargement des prédictions
                        prediction_data = raw_data.assign(Risque_Predit=probability)
                        try:
                            date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            csv = prediction_data.assign(Date_Prediction=date_str).to_csv(index=False)
                        except AttributeError:
                            st.error("Erreur lors de la prédiction : module 'datetime' has no attribute 'now'. Utilisation d'une date par défaut.")
                            date_str = "2025-07-14 00:00:00"  # Date par défaut
                            csv = prediction_data.assign(Date_Prediction=date_str).to_csv(index=False)
                        st.download_button(
                            label="📥 Télécharger les Prédictions en CSV",
                            data=csv,
                            file_name="prediction_hypertension_ml.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction : {str(e)}")
                        st.write("Détails de l'erreur :", traceback.format_exc())

        elif choice == "🧠 Prédiction Deep Learning":
            import pandas as pd
            import numpy as np
            import os
            import tensorflow as tf
            from PIL import Image
            import matplotlib.pyplot as plt
            import io
            st.markdown('<h2 style="color: #060e80; font-weight: bold; border-bottom: 2px solid #060e80; padding-bottom: 5px;">Prédiction d\'HTA par Deep Learning à partir d\'image rétiniennes</h2>', unsafe_allow_html=True)
            st.markdown('<p style="color: #34495E;">Analysez les images rétiniennes pour prédire le risque d\'HTA.</p>', unsafe_allow_html=True)

            st.warning(
                "⚠️ Estimations basées sur des images, non substitut à un diagnostic médical. Consultez un médecin."
            )

            @tf.keras.utils.register_keras_serializable(package="Custom", name="preprocess_grayscale_to_rgb")
            def preprocess_grayscale_to_rgb(image):
                return tf.image.grayscale_to_rgb(image)

            model_path = 'hypertensive_classification_model.h5'
            if not os.path.exists(model_path):
                st.error("Modèle introuvable.")
                st.stop()
            try:
                model = tf.keras.models.load_model(model_path, custom_objects={'preprocess_grayscale_to_rgb': preprocess_grayscale_to_rgb}, compile=False)
                st.success("Modèle chargé !")
            except Exception as e:
                st.error(f"Erreur modèle : {str(e)}")
                st.stop()

            st.markdown('<h4 style="color: #060e80; font-weight: bold; border-bottom: 1px solid #060e80; padding: 5px;">📸 Téléchargez une Image</h4>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Image rétinienne (PNG/JPG)", type=["png", "jpg", "jpeg"], help="Choisissez une image rétinienne.")

            if uploaded_file:
                img = Image.open(uploaded_file).convert('L')
                if img.size[0] < 100 or img.size[1] < 100:
                    st.error("Image trop petite (min 100x100).")
                    st.stop()
                img_display = img.resize((400, 400), Image.Resampling.LANCZOS)
                st.image(img_display, caption="Image Téléchargée", use_container_width=False, width=400)

                IMG_HEIGHT, IMG_WIDTH = 128, 128
                img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.Resampling.LANCZOS)
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                img_array = np.expand_dims(img_array, axis=-1)

                if st.button("🔍 Prédire"):
                    with st.spinner("Analyse..."):
                        try:
                            prediction = model.predict(img_array)
                            probability = float(prediction[0][0])
                            hta_status = "HTA" if probability > 0.5 else "Non-HTA"  # Seuil par défaut à 0.5

                            st.success("Prédiction réussie !")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(
                                    f"""
                                    <div class="prediction-card" style="background: linear-gradient(135deg, #e0e7ff 0%, #f0f2f6 100%); padding: 20px; border-radius: 15px; box-shadow: 0 6px 12px rgba(0,0,0,0.2); text-align: center; border: 3px solid #060e80;">
                                        <h3 style="color: #060e80; font-size: 2rem; margin: 0; font-weight: bold; text-transform: uppercase;">Probabilité d'HTA</h3>
                                        <p style="color: #34495E; font-size: 3rem; font-weight: bold; margin: 15px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">{probability:.2f}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                color = "#FF6B6B" if probability >= 0.7 else "#808080" if probability >= 0.5 else "#51CF66"
                                st.markdown(f'<p style="color: {color}; font-size: 1.3rem; font-weight: bold;">Statut : {hta_status}</p>', unsafe_allow_html=True)
                            with col2:
                                st.markdown(
                                    f"""
                                    <div class="prediction-card" style="background: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center; border: 2px solid #060e80;">
                                        <h3 style="color: #060e80; font-size: 1.5rem; margin: 0; font-weight: bold;">Seuil par Défaut</h3>
                                        <p style="color: #34495E; font-size: 2rem; font-weight: bold; margin: 10px 0;">0.5</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            # Vérification et affichage de la jauge Plotly
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number", value=probability, title={"text": "Risque d'HTA"},
                                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#060e80"},
                                    'steps': [{'range': [0, 0.5], 'color': "#51CF66"},
                                              {'range': [0.5, 0.7], 'color': "#808080"},
                                              {'range': [0.7, 1], 'color': "#FF6B6B"}],
                                    'threshold': {'line': {'color': "#FF6B6B", 'width': 2}, 'value': 0.7}}
                            ))
                            st.plotly_chart(fig, use_container_width=True)  # Ajusté pour s'adapter à la largeur

                            if probability < 0.5:
                                st.info("Risque faible. Mode de vie sain recommandé.")
                            elif 0.5 <= probability < 0.7:
                                st.warning("Risque modéré. Surveillez et consultez un médecin.")
                            else:
                                st.error("Risque élevé ! Consultez un médecin rapidement.")

                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            image_data = img_byte_arr.getvalue()

                            # Ajout du bouton de téléchargement des prédictions
                            prediction_data = pd.DataFrame({'Image': [uploaded_file.name], 'Probabilité d\'HTA': [probability], 'Statut Prédit': [hta_status]})
                            try:
                                date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                csv = prediction_data.assign(Date_Prediction=date_str).to_csv(index=False)
                            except AttributeError:
                                st.error("Erreur lors de la prédiction : module 'datetime' has no attribute 'now'. Utilisation d'une date par défaut.")
                                date_str = "2025-07-14 00:00:00"  # Date par défaut
                                csv = prediction_data.assign(Date_Prediction=date_str).to_csv(index=False)
                            st.download_button(
                                label="📥 Télécharger les Prédictions en CSV",
                                data=csv,
                                file_name="prediction_hypertension_deep.csv",
                                mime="text/csv"
                            )

                        except Exception as e:
                            st.error(f"Erreur lors de la prédiction : {str(e)}")
                            st.write("Détails de l'erreur :", traceback.format_exc())

                    st.markdown('<p style="color: #34495E;">Image en haute résolution requise pour une prédiction précise.</p>', unsafe_allow_html=True)

        # Section Chat Médical
        elif choice == "🤖 Chat Médical":
            st.title(f"🤖 Chatbot du GMSH - Assistant IA Génératif avec Groq")
            st.image("./bot3.png", caption="Assistant IA Groq", width=400)
            st.markdown("🧠 Posez vos questions en langage naturel. Le modèle Groq répond avec intelligence, et vous pouvez écouter la réponse.")

            # Affichage de l'historique dans la sidebar
            with st.sidebar:
                st.markdown("### Historique de la Conversation")
                if "chatgen_history" in st.session_state and st.session_state.chatgen_history:
                    for i, entry in enumerate(st.session_state.chatgen_history):
                        role = "Assistant" if entry["role"] == "assistant" else "Utilisateur"
                        st.markdown(f"**{role} ({i+1}):** {entry['content']}")
                else:
                    st.markdown("Aucun historique disponible.")

            # Gestion de l'historique du chat
            if "chatgen_history" not in st.session_state:
                st.session_state.chatgen_history = []

            for entry in st.session_state.chatgen_history:
                role = entry["role"]
                avatar = "🛡" if role == "assistant" else "👤"
                with st.chat_message(role, avatar=avatar):
                    st.markdown(entry["content"])

            # Entrée de l'utilisateur
            prompt = st.chat_input("Posez ta question sur la prédiction de l'hypertension...")
            if prompt and prompt.strip():  # Vérification que prompt n'est pas vide
                st.chat_message("user", avatar="👤").markdown(prompt)
                st.session_state.chatgen_history.append({"role": "user", "content": prompt})

                with st.spinner("💬 Génération de la réponse..."):
                    try:
                        # Utilisation directe de groq_client
                        response = groq_client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "Tu es un assistant IA expert en hypertension artérielle et santé cardiovasculaire."},
                                {"role": "user", "content": prompt}
                            ],
                            model="meta-llama/llama-4-scout-17b-16e-instruct",
                            max_tokens=512,
                            temperature=0.7
                        )
                        reply = response.choices[0].message.content
                        if reply:
                            st.chat_message("assistant", avatar="🛡").markdown(reply)
                            st.session_state.chatgen_history.append({"role": "assistant", "content": reply})
                            # Génération et lecture de l'audio avec gTTS
                            audio_content = gtts_tts(reply)
                            if audio_content:
                                st.audio(audio_content, format="audio/wav")
                                st.markdown("""
                                    <style>
                                        #audio-status {
                                            display: none; /* Masqué par défaut */
                                        }
                                        .audio-playing {
                                            display: inline-block;
                                            animation: pulse 1.5s infinite;
                                            color: #2ECC71;
                                            font-size: 20px;
                                        }
                                        @keyframes pulse {
                                            0% { transform: scale(1); }
                                            50% { transform: scale(1.1); }
                                            100% { transform: scale(1); }
                                        }
                                        .audio-error {
                                            color: #FF4136;
                                            font-size: 18px;
                                        }
                                    </style>
                                    <div id="audio-status" class="audio-playing">🎙️ Lecture en cours...</div>
                                    <script>
                                        const audio = document.querySelector('audio');
                                        const statusDiv = document.getElementById('audio-status');
                                        if (audio) {
                                            statusDiv.style.display = 'inline-block'; // Afficher uniquement quand audio existe
                                            audio.play().catch(error => {
                                                console.log('Lecture automatique bloquée :', error);
                                                statusDiv.style.display = 'none'; // Masquer en cas d'erreur
                                                st.error('Lecture bloquée - Cliquez sur le lecteur.');
                                            });
                                            audio.onended = () => {
                                                statusDiv.style.display = 'none'; // Masquer quand la lecture est terminée
                                            };
                                        } else {
                                            statusDiv.style.display = 'none'; // Masquer si aucun audio
                                        }
                                    </script>
                                """, unsafe_allow_html=True)
                            else:
                                st.error("Débogage - Aucun audio généré. Vérifiez votre connexion ou installez gTTS avec 'pip install gTTS'.")
                    except Exception as e:
                        st.error(f"🚨 Erreur lors de la génération de la réponse : {e}")

            # Boutons en bas
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:  # Bouton centré pour effacer
                if st.button("🗑 Effacer l'Historique et la Discussion"):
                    st.session_state.chatgen_history = []
                    st.rerun()
            with col3:
                if st.button("🔊 Écouter la réponse") and st.session_state.chatgen_history and st.session_state.chatgen_history[-1]["role"] == "assistant":
                    try:
                        latest_reply = st.session_state.chatgen_history[-1]["content"]
                        audio_content = gtts_tts(latest_reply)
                        if audio_content:
                            st.audio(audio_content, format="audio/wav")
                            st.markdown("""
                                <style>
                                    #audio-status {
                                        display: none; /* Masqué par défaut */
                                    }
                                    .audio-playing {
                                        display: inline-block;
                                        animation: pulse 1.5s infinite;
                                        color: #2ECC71;
                                        font-size: 20px;
                                    }
                                    @keyframes pulse {
                                        0% { transform: scale(1); }
                                        50% { transform: scale(1.1); }
                                        100% { transform: scale(1); }
                                    }
                                    .audio-error {
                                        color: #FF4136;
                                        font-size: 18px;
                                    }
                                </style>
                                <div id="audio-status" class="audio-playing">🎙️ Lecture en cours...</div>
                                <script>
                                    const audio = document.querySelector('audio');
                                    const statusDiv = document.getElementById('audio-status');
                                    if (audio) {
                                        statusDiv.style.display = 'inline-block'; // Afficher uniquement quand audio existe
                                        audio.play().catch(error => {
                                            console.log('Lecture automatique bloquée :', error);
                                            statusDiv.style.display = 'none'; // Masquer en cas d'erreur
                                            st.error('Lecture bloquée - Cliquez sur le lecteur.');
                                        });
                                        audio.onended = () => {
                                            statusDiv.style.display = 'none'; // Masquer quand la lecture est terminée
                                        };
                                    } else {
                                        statusDiv.style.display = 'none'; // Masquer si aucun audio
                                    }
                                </script>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"🚨 Erreur avec gTTS : {e}. Vérifiez votre connexion ou installez gTTS avec 'pip install gTTS'.")

    st.markdown("""
        <div class="footer">
            <p>© 2025 Amougou André - Groupe Médical St-Hilaire | <a href="mailto:contact@gmsh.fr" style="color: #FFFFFF;">Contact</a></p>
        </div>
    """, unsafe_allow_html=True)

    # Ajustement du padding pour éviter le chevauchement avec le footer
    st.markdown("""
           <script>
            const footer = document.querySelector('.footer');
            const main = document.querySelector('.main-container');
            if (footer && main) {
                main.style.paddingBottom = footer.offsetHeight + 'px';
            }
        </script>
    """, unsafe_allow_html=True)
            
if __name__ == "__main__":
    main()