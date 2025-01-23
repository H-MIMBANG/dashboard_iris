#Importation des bibliothèques nécessaires
import streamlit as st
import requests
import pandas as pd
import matplotlib as plt
import seaborn as sns

# Titre de l'application
st.title("Dashboard de prédiction avec le modèle Random Forest")

# Formulaire pour saisir les données d'entrée
st.header("Saisir les caractéristiques pour la prédiction")

# Chargement des données
df = pd.read_csv('Iris.csv',delimiter=';')

# Histogramme des longueurs des sépales
st.subheader('Histogramme de la longueur des sépales')
plt.figure(figsize=(10,5))
sns.histplot(df['SepalLength'], bins=20, kde=True)
st.pyplot(plt)

# About

if st.button("About App"):
	st.subheader("App d'exploration des données d'Iris")
	st.text("Construite avec Streamlit")
	st.text("Thanks to the Streamlit Team Amazing Work")

if st.checkbox("By"):
	st.text("Heddy MIMBANG")
	st.text("heddy.mimbang@icloud.com")
