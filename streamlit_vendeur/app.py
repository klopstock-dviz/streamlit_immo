import streamlit as st
from streamlit_folium import folium_static, st_folium
from data import generate_description, get_plots
import plotly.express as px
import os
from PIL import Image
import matplotlib.pyplot as plt


# Streamlit app
st.title('Générateur de descriptions immobilières')

# User inputs
adresse = st.text_input('Adresse du bien', value='57 quai georges gorce, boulogne b')

st.markdown('<hr>', unsafe_allow_html=True)

# Création de deux colonnes pour les champs "type_bien_vendeur" et "type_transaction"
col1, col2 = st.columns(2)
with col1:
    type_bien_vendeur = st.selectbox('Type de bien', options=['Maison/Villa', 'Appartement'], index=1)
with col2:
    type_transaction = st.selectbox('Type de transaction', options=['Vente', 'Location'], index=0)

st.markdown('<hr>', unsafe_allow_html=True)

# Création de trois colonnes pour les champs "nb_pieces", "surface" et "prix"
col3, col4, col5 = st.columns(3)
with col3:
    nb_pieces = st.number_input('Nombre de pièces', min_value=1, value=3)
with col4:
    surface = st.number_input('Surface (m²)', min_value=9, value=70)
with col5:
    prix = st.number_input('Prix (€)', min_value=100, value=500000)

if st.button('Générer la description'):
    # Display the generated description
    st.subheader('Description du bien')
    response_stream = generate_description(adresse, type_bien_vendeur, type_transaction, nb_pieces, surface, prix)

    # Zone pour afficher la réponse en streaming
    response_placeholder = st.empty()
    full_response = ""

    # Boucle pour afficher la réponse au fur et à mesure
    for chunk in response_stream:
        full_response += chunk  # Ajoute chaque morceau de réponse
        response_placeholder.markdown(full_response)  # Met à jour l'affichage    

    st.markdown("### Images du bien:", unsafe_allow_html=True)    
    st.image("./galerie_3x3.png",)    
    
    st.markdown(f'----', unsafe_allow_html=True)

    plots=get_plots()
    _map=plots[0]
    legend_map=plots[1]
    data_prix=plots[2]

    # Charger Font Awesome
    st.markdown(
        '''
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        
        ''',
        unsafe_allow_html=True
    )
    st.markdown(f'----', unsafe_allow_html=True)
    st.markdown(f"### Commerces et services:", unsafe_allow_html=True)
    folium_static(_map, width=1000, height=700)
    
    st.markdown(f'----', unsafe_allow_html=True)
    st.markdown(f"### Relevé des prix:", unsafe_allow_html=True)
    # Couleurs personnalisées
    gen_color="#4e7dbb"
    colors = [gen_color, gen_color, gen_color, "#FF0000"]

    # Créer un graphique avec plotly
    fig = px.bar(data_prix, x="label", y="value", labels={"value": "Valeur", "label": "Label"})
    fig.update_traces(marker_color=colors)


    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)