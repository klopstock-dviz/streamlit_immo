import streamlit as st
import asyncio
import nest_asyncio
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path
import joblib
import folium
from streamlit_folium import folium_static, st_folium
from folium.plugins import MeasureControl
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
from Levenshtein import ratio as levenshtein_ratio
import copy
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv
import plotly.express as px
from plotly.subplots import make_subplots


# Patch torch.classes pour éviter l'erreur de Streamlit
import torch



#=========corrections incompatibilités streamlit / event loop
#=============== nécessaire pour streaming des réponses graphRAG
# Patch the event loop to allow nested async calls
nest_asyncio.apply()
#==========fin

#=================== Correction conflit event loops torch/streamlit
# Save the original __getattr__ method
original_getattr = torch._classes._Classes.__getattr__

# Define a patched version to handle __path__
def patched_getattr(self, attr):
    if attr == "__path__":
        # Explicitly block access to __path__
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '__path__'")
    return original_getattr(self, attr)

# Apply the patch
torch._classes._Classes.__getattr__ = patched_getattr
#===============fin

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)
client = OpenAI()

# Set page config
st.set_page_config(layout="wide", page_title="Real Estate Search Assistant")

# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent.resolve()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(SCRIPT_DIR/"data/df_desciption_photos.csv")
    # df = pd.read_csv(SCRIPT_DIR/"data/df_desciptions.csv")
    
    cols = ['typedebien', 'typedetransaction', 'ville', 'nomQuartier', 'etage', 'surface', 
            'nb_pieces', 'prix_bien', 'description_bien', 'nb_etages', 'parking', 
            "annee_construction", "nb_logements_copro", 'prix_m_carre', 'chauffage_energie',
            'code_dep', 'resume_fr']
    
    df.dropna(subset=["ville", "prix_bien", "nb_pieces", "surface", "prix_m_carre"], inplace=True)
    df = df.reset_index(drop=True)
    
    # Load embeddings
    desc_embeddings = joblib.load(SCRIPT_DIR/"data/desc_embeddings.joblib")
    resume_embeddings = joblib.load(SCRIPT_DIR/"data/resume_embeddings.joblib")
    
    return df, desc_embeddings, resume_embeddings

df, desc_embeddings, resume_embeddings = load_data()

# Initialize models
@st.cache_resource
def load_models():
    model_name = 'dangvantuan/sentence-camembert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_emb = AutoModel.from_pretrained(model_name)
    return tokenizer, model_emb

tokenizer, model_emb = load_models()

# Helper functions (from your original code)
def generate_embeddings(texts, max_length=512):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    with torch.no_grad():
        outputs = model_emb(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

def extract_json(response_content):
    if response_content.find("```json") > -1:
        start = response_content.find("```json") + 7
        end = response_content.find("```", start + 1)
        json_resp = response_content[start: end]
        try:
            return json.loads(json_resp)    
        except Exception as e:
            print(f"error on extract json: {e}")
            return None
    else:
        try:
            return json.loads(response_content)
        except Exception as e:
            print(f"error convertion json: {e}")
            return None

def get_coords_ville(ville):
    headers = {'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8'}
    call_ors_status_code, call_datagouv_status_code=(0,0)
    try:
        call_ors = requests.get(f'https://api.openrouteservice.org/geocode/search?api_key=5b3ce3597851110001cf624880dfd5c82fff4542977ea0486794cb08&text={ville},%20France', headers=headers)    
        call_ors_status_code=call_ors.status_code
    except Exception as e:
        print(e)

    try:
        call_datagouv = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={ville}&limit=5")
        call_datagouv_status_code=call_datagouv.status_code
    except Exception as e:
        print(e)
    
    if call_ors_status_code == 200:
        geo_resp = json.loads(call_ors.text)
        lat = geo_resp["features"][0]["geometry"]["coordinates"][1]
        lon = geo_resp["features"][0]["geometry"]["coordinates"][0]
        region = geo_resp["features"][0]["properties"]["region"]
        return {'lat': lat, "lon": lon, "region": region}
    elif call_datagouv_status_code == 200:
        geo_resp = json.loads(call_datagouv.text)
        lat = geo_resp["features"][0]["geometry"]["coordinates"][1]
        lon = geo_resp["features"][0]["geometry"]["coordinates"][0]
        region = geo_resp["features"][0]["properties"]["context"]
        return {'lat': lat, "lon": lon, "region": region}

def calc_distance_km(point1, point2):
    distance = geodesic(point1, point2).km
    return distance

def build_filter_query(criteria, distance_proximite_km=20):
    criteria = copy.deepcopy(criteria)
    cols = ['typedebien', 'typedetransaction', 'ville', 'nomQuartier', 'etage', 'surface', 
            'nb_pieces', 'prix_bien', 'description_bien', 'nb_etages', 'parking', 
            "annee_construction", "nb_logements_copro", 'prix_m_carre', 'chauffage_energie',
            'code_dep', 'resume_fr']

    if ("ville" in criteria) and ("recherche_type" in criteria):
        coords = get_coords_ville(criteria["ville"])
        if coords:
            lat = coords["lat"]
            lon = coords["lon"]
            coords_ref_point = (lat, lon)
            df['distance_km'] = df.apply(lambda row: np.round(calc_distance_km(
                coords_ref_point, (row['mapCoordonneesLatitude'], row['mapCoordonneesLongitude'])), 1), axis=1)

            if (criteria["recherche_type"] == "proximité"):
                del criteria["ville"]

    string_cols = df.select_dtypes(include=['object']).columns.intersection(cols)
    numeric_cols = df.select_dtypes(include=['number']).columns.intersection(cols)

    conditions = []

    for key, value in criteria.items():
        if key in cols + ["recherche_type"]:
            if isinstance(value, dict):
                min_val = value.get('min')
                max_val = value.get('max')
                if min_val is not None:
                    if isinstance(min_val, str) and min_val.lower() == 'null':
                        min_val = None
                    if min_val is not None:
                        conditions.append(f'`{key}` >= {min_val}')
                if max_val is not None:
                    if isinstance(max_val, str) and max_val.lower() == 'null':
                        max_val = None
                    if max_val is not None:
                        conditions.append(f'`{key}` <= {max_val}')
            else:
                if isinstance(value, str) and value.lower() == 'null':
                    value = None
                if value is not None:
                    if isinstance(value, str):
                        if key == "ville":
                            match_villes = []
                            for ville in df["ville"].unique():
                                match_villes.append({"ville": ville, "ville_match": levenshtein_ratio(value, ville)})
                            result = pd.DataFrame(match_villes).sort_values(by="ville_match", ascending=False).head(1)["ville"]
                            if len(result) > 0:                            
                                conditions.append(f'`{key}`.str.startswith("{result.values[0]}")')
                        elif key == "typedebien" and value == "Maison":
                            value += "/Villa"
                            conditions.append(f'`{key}` == "{value}"')
                        elif key == "recherche_type" and value == "proximité":
                            conditions.append(f'`distance_km` <= {distance_proximite_km}')
                        elif key != "recherche_type":
                            conditions.append(f'`{key}` == "{value}"')
                    else:
                        conditions.append(f'`{key}` == {value}')

    if conditions:
        query = ' & '.join(conditions)
    else:
        query = None
    
    return query

def plot_map(filtered_df, criteria):
    coords_ref_point = None
    if "ville" in criteria:
        coords = get_coords_ville(criteria["ville"])
        if coords:
            lat = coords["lat"]
            lon = coords["lon"]
            coords_ref_point = (lat, lon)
    
    if coords_ref_point:
        m = folium.Map(
            location=coords_ref_point, 
            zoom_start=11,    
            width='100%',        # prend 100% du conteneur parent
            height=600
        )

        MeasureControl().add_to(m)
        folium.Marker(
            location=coords_ref_point,
            popup="Ville de référence",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
    else:
        random_point = filtered_df.sample(1)[["mapCoordonneesLatitude","mapCoordonneesLongitude"]].values
        random_point = (random_point[0][0], random_point[0][1])
        m = folium.Map(
            location=random_point, 
            zoom_start=5,
            width='100%',        # prend 100% du conteneur parent
            height=600
        )            

    for index, row in filtered_df.iterrows():
        folium.Marker(
            location=(row["mapCoordonneesLatitude"], row["mapCoordonneesLongitude"]),
            popup=folium.Popup(
                f"""
                Ville: {row['ville']}<br>
                Quartier: {row['nomQuartier']}<br>
                Bien: {row["typedebien"]}<br>
                {row['nb_pieces']} pièces<br>
                {row['surface']} m²<br>
                {row['prix_bien']} €<br>
                {np.round(row["distance_km"],1)} km du centre
                """,
                max_width=200
            ),
            icon=folium.Icon(color="blue")
        ).add_to(m)

    return m



def create_interactive_charts(filtered_df):
    # 1) Création de la figure 2×2
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Nombre de biens par ville",
            "Distribution des surfaces",
            "Prix moyen des biens par ville",
            "Prix moyen au mètre carré (Top 5 des villes)"
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.12
    )

    # ——————————————————————————
    # 2) Bar chart : Nombre de biens par ville (top 5)
    # ——————————————————————————
    ville_counts = filtered_df['ville'].value_counts().nlargest(5)
    bar_trace = px.bar(
        x=ville_counts.index,
        y=ville_counts.values,
        labels={'x': 'Ville', 'y': 'Nombre de biens'}
    ).data[0]

    # Optionnel : ajouter une bordure pour plus de contraste
    bar_trace.update(marker_line_color="white", marker_line_width=2)

    fig.add_trace(bar_trace, row=1, col=1)

    # ——————————————————————————
    # 3) Histogramme : Distribution des surfaces
    # ——————————————————————————
    hist_trace = px.histogram(
        filtered_df,
        x='surface',
        #nbins=30,
        labels={'surface': 'Surface (m²)', 'count': 'Effectif'}
    ).data[0]

    # On supprime la bordure pour donner l'impression de barres collées
    hist_trace.update(marker_line_width=0)

    fig.add_trace(hist_trace, row=1, col=2)

    # ——————————————————————————
    # 4) Préparation top 5 pour box‐plots
    # ——————————————————————————
    top_5_villes = ville_counts.index
    df_top5 = filtered_df[filtered_df['ville'].isin(top_5_villes)]

    # 5) Box‐plot prix du bien
    box1 = px.box(
        df_top5,
        x='ville',
        y='prix_bien',
        labels={'ville': 'Ville', 'prix_bien': 'Prix du bien (€)'}
    ).data[0]
    fig.add_trace(box1, row=2, col=1)

    # 6) Box‐plot prix au m²
    box2 = px.box(
        df_top5,
        x='ville',
        y='prix_m_carre',
        labels={'ville': 'Ville', 'prix_m_carre': 'Prix au m² (€)'}
    ).data[0]
    fig.add_trace(box2, row=2, col=2)

    # ——————————————————————————
    # 7) Mise à jour de la mise en page globale
    # ——————————————————————————
    fig.update_layout(
        height=800,
        width=1000,
        showlegend=False,
        title_text="Analyse interactive des annonces immobilières",
        bargap=0.4  # augmente l'espace entre les barres 'bar' (graf 1)
    )

    # Rotation des labels
    for i in range(1, 5):
        fig['layout'][f'xaxis{i}'].update(tickangle=45)

    return fig


def extract_structured_data(user_query):
    def get_geosearch_type(user_query):
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "tu es un assitant qui aide à déterminer si une recherche d'annonces est demandée dans une ville précise, ou à proximité de celle ci"
                },
                {
                    'role': 'user',
                    'content': f"""Dans la requête ci-dessous, la recherche porte-t-elle sur:\n 
                        * une ville précise (retourne "exact" si oui) ? Observe la présence de termes comme "à", "dans", "sur" \n 
                        * une zone approximative (retourne "proximité" si oui) ? Observe la présence de termes comme "autour", "à proximité", "dans la zone/région", "sur la zone/région" \n 
                        Requête:\n {user_query}.\n
                        Réponds uniquement avec un objet JSON strictement valide, avec des guillemets doubles : {{ "recherche_type": "..." }}
                    """
                }
            ],
            model="gpt-4",
            temperature=0.1,
        )

        response_content = response.choices[0].message.content
        try:
            clean_response_content = response_content.replace("```json", '').replace("```",'').replace("\n", "").replace("'", '"')
            return json.loads(clean_response_content)
        except Exception as e:
            print(e)
            return None

    prompt_v1 = f"""
        Extrait les informations suivantes de la requête utilisateur au format JSON :
        - ville: (Poissy, Paris, Nice ...)
        - typedebien (Maison/Villa, Appartement)
        - typedetransaction (Vente, Location)
        - nb_pieces (entier représentant le nombre de pieces) 
        - surface 
        - prix_bien (prix du logement)
        - prix_m_carre (prix au mettre carré (m2) du logement)
        - autres_criteres (liste de chaînes de caractères)

        Quand la requête mentionne un critère numérique avec des bornes du genre "au moins", "minimum" ou "maximum", il faut renvoyer un dictionnaire comportant ces bornes
        Exemples:
        * Exemple 1 de prix: Demande "prix au maximum 300000": renvoi {{"prix": {{"min": null, "max": 300000}}}}
        * Exemple 2 de prix: Demande "prix de 200000 jusqu'à 300000": renvoi {{"prix": {{"min": 200000, "max": 300000}}}}
        
        Quand d'autres critères d'équipements et services sont mentionnés, renvoi les sous forme de liste
        Exemple:
        "autres_criteres": ["balcon", "wc", "séjour lumineux" ...]
        Ceci est la requête utilisateur : "{user_query}"

        Ta sortie doit être limitée au json demandé ci-dessus
    """

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "tu es un assitant qui extrait des informations structurées de requêtes utilisateurs pour des recherches immobilières"
            },
            {
                'role': 'user',
                'content': prompt_v1
            }
        ],
        model="gpt-4",
        temperature=0.1,
    )

    response_content = response.choices[0].message.content
    try:
        user_data = extract_json(response_content)
        ind_search = get_geosearch_type(user_query)
        if ind_search is not None:
            user_data.update(ind_search)
        return user_data, response_content
    except Exception as e:
        print(e)
        return None, response_content

def generate_rag_prompt(user_query, filtered_df, target_text="description_bien"):
    query_embedding = generate_embeddings([user_query])
    filtered_indices = filtered_df.index.tolist()
    filtered_desc_embeddings = desc_embeddings[filtered_indices]
    filtered_resume_embeddings = resume_embeddings[filtered_indices]

    similarities_desc = cosine_similarity(query_embedding, filtered_desc_embeddings)
    similarities_resume = cosine_similarity(query_embedding, filtered_resume_embeddings)

    tableau_general_similarites = []
    cols_for_llm = ['typedebien', 'ville', 'nomQuartier', 'surface', 'nb_pieces', 
                   'prix_bien', 'prix_m_carre', 'distance_km']

    for a, i in enumerate(filtered_indices):
        tableau_general_similarites.append({
            "i": i,
            "description_bien": f'Fiche du bien:\n {filtered_df.loc[i, cols_for_llm].to_dict()}\n\nDescription:\n{filtered_df.loc[i, "description_bien"]}',
            "similarity_description_bien": similarities_desc[0][a],
            "resume_bien": filtered_df.loc[i, "resume_fr"],
            "similarity_resume_bien": similarities_resume[0][a],        
        })

    df_tableau_general_similarites = pd.DataFrame(tableau_general_similarites)

    champs = {
        "description_bien": {"text": "description_bien", "similarity": "similarity_description_bien"},
        "resume_bien": {"text": "resume_bien", "similarity": "similarity_resume_bien"}
    }

    top_desc_bien = df_tableau_general_similarites.sort_values(by=champs[target_text]["similarity"], ascending=False).head(4)
    
    ensemble_annonces = ""
    for num_annonce, idx in enumerate(top_desc_bien.index, 1):
        ensemble_annonces += f'Annonce {num_annonce}:\n{top_desc_bien.loc[idx, champs[target_text]["text"]]}\n-------------------\n\n'

    return ensemble_annonces

# Streamlit UI
st.title("Compagnon de recherche immobilière")

# Query input
user_query = st.text_area("Saisir votre recherche:", 
                         "Je cherche un appartement (vente), 3 pièces et 55m2 minimum, à 600 000€ max, autour de Boulogne 92, avec balcon et séjour lumineux, parking obligatoire")

if st.button("Search"):
    with st.spinner("Processing your query..."):
        # Extract structured data
        json_resp, raw_response = extract_structured_data(user_query)
        
        if json_resp:
            st.subheader("Extraction des critères de recherche")
            st.json(json_resp)
            
            # Build and apply filter
            pandas_query = build_filter_query(json_resp)
            if pandas_query:
                filtered_df = df.query(pandas_query)#.sample(200)
                
                if not filtered_df.empty:
                    st.success(f"Found {len(filtered_df)} matching properties")
                    
                    # Display stats and map
                    col1, col2 = st.columns(2)
                    
                    #with col1:
                    st.subheader("Statistiques")
                    stats_fig = create_interactive_charts(filtered_df)
                    if stats_fig:
                        st.plotly_chart(stats_fig, use_container_width=True)

                
                    #with col2:
                    st.subheader("Carte des biens")
                    m = plot_map(filtered_df, json_resp)

                    folium_static(m, width=1280, height=800)

                
                    # Generate RAG prompt and get LLM response
                    st.subheader("Recommendations")
                    with st.spinner("Analyse des biens..."):
                        ensemble_annonces = generate_rag_prompt(user_query, filtered_df)
                        prompt = f"""
                        Au regard de la requête utilisateur :
                        "{user_query}"

                        Il est possible d'évaluer et comparer les annonces suivantes :
                        {ensemble_annonces}

                        Merci de proposer les meilleures annonces sous forme de **tableau Markdown**, avec :
                        - Les annonces en **colonnes** (numéro, localisation, surface, prix, etc.),
                        - Les critères en **lignes** (prix, surface, distance, luminosité, terrasse/balcon, parking, résidence sécurisée, correspondance budget, recommandation finale).

                        """
                        messages = [
                            {"role": "system", "content": "Tu es un assistant immobilier, tu vas conseiller les acheteurs sur les annonces qui correspondent le mieux à leurs demandes"},
                            {"role": "user",   "content": prompt}
                        ]                        

                        container = st.empty()
                        full_response = ""

                        # --- appel synchrone avec stream=True ---
                        response_stream = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            temperature=0.1,
                            stream=True,
                        )

                        # --- itération sur le générateur de chunks ---
                        for chunk in response_stream:
                            # chunk est un objet pydantic ; on extrait le texte s’il existe
                            content = getattr(chunk.choices[0].delta, "content", None)
                            if content:
                                full_response += content
                                container.markdown(full_response)  # mise à jour progressive

                        st.success("Réponse complète générée.")

                else:
                    st.warning("No properties found matching your criteria")
            else:
                st.error("Could not build a valid search query")
        else:
            st.error("Could not extract search criteria from your query")
            st.text(raw_response)