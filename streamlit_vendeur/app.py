import streamlit as st
from streamlit_folium import folium_static
from st_keyup import st_keyup
import requests
from data import generate_description, get_map_poi, get_chart_releve_prix, get_chart_histo_prix, get_charts_histo_activity
from pathlib import Path

st.set_page_config(
    page_title="Agent de vente IA",
    layout="centered",
    page_icon="./icons/house-chimney-solid.svg"
)

# Charger Font Awesome
st.markdown(
    '''
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    
    ''',
    unsafe_allow_html=True
)

# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent.resolve()
st.markdown("<span style='font-size: 2rem; display: flex; justify-self: centrer'>Assistant de rédaction et d’estimation immobilière</span>", unsafe_allow_html=True)
    
st.image("./icons/header.png")

# Streamlit app
st.markdown("""
    ---
    Générer des descriptions pour votre annonce immobilière à partir des photos de votre bien et de son emplacement.
            
    Il vous sera proposé: 
    1. Une trame pour votre annonce (description du bien et son emplacement) 
    2. Des conseils portant sur le prix que vous demandez, basés sur:
        * les biens similaires observés sur le marché au cours des 4 derniers mois.
        * l'évolution des prix de marché sur la dernière année.
        * l'évolution du nombre de biens disponibles sur la dernière année.
        * la durée moyenne de validité des annonces sur la dernière année.
    """, unsafe_allow_html=True)

st.write("---")

st.markdown("### Plus d'nformations sur cette démo:")
with st.expander("Lire ...", expanded=0):
    st.markdown(f"""
                    
        #### Objectif de cette application:
        Générer une description de vente/location d'un logement, en exploitant les photos du bien avec des méthodes image-to-text.<br>
        La description ainsi produite se veut plus neutre et factuelle que des annonces commerciales classiques.<br>
                    
        Cette description est la synthèse des sources suivantes:
        1. Données non structurées: <br>
            * photos du bien à vendre/louer <sup>1</sup><br>
        
        >-> Sélection automatique ou fourni par l'utilisateur
        
        2. Données structurées du bien:
            * Adresse
            * Type de bien
            * Taille du bien
            * Prix de vente ou loyer demandé

        >-> Fourni par l'utilisateur
                    
        3. Données structurées du quartier/commune/département:
            * Données démographiques et socio-économiques:
                * Population (individus et ménages)
                * Diplômes
                * Chômage
                * Revenus
                * Pauvreté
            * Commerces et services à proximité
            * Prix des biens similaires <sup>2</sup>
        <br>
        >-> Déduit de l'adresse du bien
                    
        #### Originalité:
        <u>**L'innovation et le défi**</u> dans ce projet est la transformation des photos en source de données valorisables pour la description.
        Pour cette démo un appel direct à un LLM multimodal avec prompt + photo permet d'obtenir une description.<br>
        Pour une haute fidélité les modèles frontière sont nécessaires (gpt 5, gemini pro ...)<br>
        Pour une bonne fidélité, avec de rares erreurs spatiales (voir plus bas) les LLM distillés des modèles frontière font un bon travail (gpt 5 mini, gemini flash ...), ainsi que des LLM multimodaux (visual LLM) open source tels LLAMA 4 Maverick ou Qwen VL (> 32b)<br>
        
        #### Ma contribution:
        Cependant si on souhaite rester dans l'open source, sur des petits LLM multimodaux (< 32b ), l'approche suivante, que j'ai développé dans le cadre de mon mémoire de formation en deep learning, donne de très bons résultats.
                    
        Cette approche suppose une étape de pré-traitement avec l'intervention de différents petits modèles de vision spécialisés, dont les résultats sont partagés avec le visual LLM cible, qui doit en fin de compte confirmer les résultats et produire son analyse finale. 
                    
        ##### Les étapes:
        1. Le modèle <a href="https://huggingface.co/openai/clip-vit-base-patch32" targe="blank">
                CLIP</a> pour catégoriser le type d'espace (hall, séjour, cuisine, jardin, vue ... )
        2. Le modèle <a href="https://huggingface.co/facebook/detr-resnet-50" targe="blank">
                DETR</a> pour détecter et énumérer les objets visibles sur la photo (vase, cheminée, lustre, piscine, mer ...)
        3. Le visual LLM, alimenté avec les sorties 1 et 2 + la photo, est chargé de confirmer le type d'espace et son contenu. A cette étape, si le LLM diverge, le risque d'hallucination est jugé élevé, la génération est stopée.
        4. Si le LLM en étape 3 converge avec les modèles CLIP et DETR, la génération de la description se poursuit.
                    
        ##### En résumé:
        Les étapes 1 et 2 permettent d'améliorer la qualité de l'interprétation, en mettant le visual LLM dans le bon contexte.<br>
        La génération en 4 étapes décrite ci dessus permet de réduire les imprécisions et les hallucinations de 20% à 10% sur un échantillon de 120 photos, avec LLAMA 3.1 11b, en référence à une analyse faite par GPT 4o, qui lui même a été évalué humainement sur la base de 50 photos, et déclaré fiable à 100%.
                    
        ##### Limites avec un serveur CPU only:
        Cependant l'analyse d'une seule image en suivant ces étapes demande 4 à 6 fois plus de temps qu'un prompt direct à un LLM, sur une cloud spécialisé.
                    
        Cette contrainte rend la méthode peu pratique pour une analyse en ligne comme proposé ici, si on ne dispose pas d'un serveur équipé d'un GPU spécialisé.
        <hr>
                            
        #### Progrès en image-to-text depuis ces travaux:
        Aujourd'hui (novembre 2025), une nouvelle génération de LLM open source de petite taille (Qwen3 VL 32B, Gemma 3 27b, Mistral small 24b) donnent de meilleurs résultats sur ce genre tâche, en suivant le pré processing évoqué plus haut.
                    
        Sur ces petits modèles, une nouvelle approche d'ajustement (<a href="https://arxiv.org/pdf/2508.19652" target="blank">Vision SR1</a>) 
        basée sur de l'<a href="https://github.com/opendilab/awesome-RLVR" taget="blank">apprentissage par renforcement</a> pour la vision semble donner des résultats positifs, avec notamment un travail de fond sur la génération d'un grand dataset réalisé par les chercheurs, ouvrant la voie vers d'avantage de fine tuning (plus grands modèles, ou tâches spécifiques).
                    
        D'autres modèles open source plus grands tels que LLAMA 4 Maverick, Qwen VL (235b, Plus), ou fermés tels que Gemini flash ou Gpt5 mini donnent de très bons résultats, pour un prix très contenu (< 1€ / 1000 images).<br>
        La limite affectant occasionnellement tous ces modèles sera une mauvaise lecture de l'effet parallaxe, et une représentation spatiale parfois défaillante, du à l'angle de certaines prises.
        
        <hr>
                                
        <sup>**1**</sup> **Choix des photos:** 
        Si le visiteur de cette page n'a pas à disposition les photos d'un bien, 
        il est proposé une sélection aléatoire des photos d'un bien correspondant
        aux critères `type de bien` et `taille du bien` saisis, sur la base de 10k photos de biens téléchargées entre novembre et décembre 2024.
                    
        <sup>**2**</sup> **Base de prix:** 
        Les prix sont issus des annonces publiées par les principales agences immobilières.    
    """, unsafe_allow_html=1)

st.write("---")

with st.container(border=1, width=720, horizontal_alignment="center"):
    # User inputs
    # adresse = st.text_input('Adresse du bien', value='57 quai georges gorce, boulogne b')

    


    API_URL = "https://api-adresse.data.gouv.fr/search/"

    @st.cache_data(ttl=300)
    def get_suggestions(query: str, limit: int = 5) -> list[dict]:
        params = {"q": query, "limit": limit, "autocomplete": 1}
        
        suggestions = []
        try:
            resp = requests.get(API_URL, params=params, timeout=5)
            # resp.raise_for_status()
            data = resp.json()
            
            for feature in data.get("features", []):
                props = feature["properties"]
                lon, lat = feature["geometry"]["coordinates"]
                suggestions.append({
                    "label": props["label"],
                    "postcode": props.get("postcode"),
                    "city": props.get("city"),
                    "lat": lat,
                    "lon": lon,
                })
        except Exception as e:
            print("Err autocompletion adresse:", e)
            
        return suggestions

    st.markdown("#### Commencer")

    # Remplace st.text_input par st_keyup pour capter chaque frappe
    query = st_keyup("Votre adresse, saisir pour afficher les propositions", value="42 quai georges", debounce=300, key="adresse")

    if query:  # dès qu'il y a du texte
        suggestions = get_suggestions(query)
        if suggestions:
            choix = st.selectbox("Sélectionnez une proposition", [s["label"] for s in suggestions])
            st.session_state["adresse"]=choix
            sel = next(s for s in suggestions if s["label"] == choix)
            # st.markdown(f"**{sel['label']}** — {sel['postcode']} {sel['city']}")
            # st.write(f"Coords : ({sel['lat']:.6f}, {sel['lon']:.6f})")
        else:
            st.info("Aucune suggestion trouvée, essayez d'affiner votre saisie.")

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

    st.markdown("---")
    expander_images=1 if "images_uploaded" in st.session_state else 0
    with st.expander(label="Cliquer pour chargez vos images ci dessous ou laissez l'app vous proposer un échantillon", expanded=expander_images):
        images_uploaded=st.file_uploader(label="Charger vos photos (optionnel)", accept_multiple_files=True, type=["jpg", 'jpeg', "png"])
        st.session_state["images_uploaded"]=1

    # espace de génération
    st.markdown("---")
    with st.container(border=1):
        st.markdown("""
            ##### Comment la génération va t elle se faire ?
        """)
        with st.expander(label="Voir les étapes"):
            st.markdown("""
                1. Chargement des photos et conversion en galerie (3*3)
                2. Utilisation d'un visual LLM pour extraire une description détaillée de cette galerie
                3. Récupération des données INSEE concernant l'emplacement du bien
                4. Récupération des données sur les commerces & services à proximité
                5. Récupération des données sur le marché immobilier
                6. Génération d'une description reprenant tous les éléments ci dessus
            """)
            # st.image("./mermaid-diagram-etapes_generation.png")

        col6, col7, col8=st.columns(3, gap="small", vertical_alignment="bottom")

        with col6:
            btn_generer_description=st.button('Générer la description', type="primary", key="btn_generer_description")
        with col7:
            if images_uploaded !=None and  len(images_uploaded)>0:
                list_models=["GPT 5 mini",]
            else:
                list_models=["GPT 5 mini",]

            images_vllm_engine=st.selectbox(
                label="Modèle pour les images:", 
                options=list_models, 
                index=0,
                key="images_vllm_engine"
            )

    st.write("---")

if btn_generer_description:
    st.markdown("### Images utilisées pour la description:", unsafe_allow_html=True)    
    placeholder_photos=st.container()    

    # # charger les photos si dispo
    # images_loaded=[]
    # if images is not None:
    #     for img in images:
    #         images_loaded.append(st.image(img))
    

    

    # Display the generated description
    # st.subheader('')
    # Zone pour afficher la réponse en streaming

    with st.spinner("Génération en cours..."):
        response_trame_annonce_placeholder = st.empty()
        st.write("---")
        map_poi_placeholder=st.empty()
        
        releve_prix_placeholder=st.empty()
        historique_prix_placeholder=st.empty()
        historique_activity_placeholder=st.empty()
        conseils_prix_placeholder=st.empty()
        response_conseils_prix_placeholder=st.empty()
        

        trame_annonce_full_response = "### 1. La trame pour votre annonce \n"
        trame_conseils_prix_full_response= ""

        for msg in generate_description(st.session_state["adresse"], type_bien_vendeur, type_transaction, nb_pieces, surface, images_vllm_engine, prix, images_uploaded):
            if msg["key"]=="build_photos_album" and msg["value"]=="thumbnail ready":
                with placeholder_photos:
                    st.image(SCRIPT_DIR/"galerie_3x3_small.png",)    
            elif msg["key"]=="analyse_galerie_online" and msg["status"]=="pending":
                response_trame_annonce_placeholder.markdown(msg["value"])
            elif msg["key"]=="trame_annonce_stream":
                chunk=msg["value"]
            # # Boucle pour afficher la réponse au fur et à mesure
                trame_annonce_full_response += chunk  # Ajoute chaque morceau de réponse
                response_trame_annonce_placeholder.markdown(trame_annonce_full_response, unsafe_allow_html=1)  
            elif msg["key"]=="map_poi":
                # map_poi=get_map_poi()
                map_poi= msg["value"]
                with map_poi_placeholder:
                    with st.container():
                        st.markdown(f"### Commerces et services:\n<i>Votre bien est représenté par la maison en bleu</i>", unsafe_allow_html=True)
                        folium_static(map_poi, width=1000, height=700)
            elif msg["key"]=="analyse_prix": 
                if msg["step"]=="chart_releve_prix":
                    charts_prix_static=get_chart_releve_prix()
                    prix_static=charts_prix_static[0]
                    prix_m2_static=charts_prix_static[1]
                    st.write("---")
                    with releve_prix_placeholder:                        
                        with st.container():
                            st.markdown(f"### 2. Analyse du prix demandé et conseils\n", unsafe_allow_html=True)
                            st.write(f"#### Relevé des prix:", unsafe_allow_html=True)
                            # Afficher le graphique dans Streamlit
                            st.plotly_chart(prix_static)
                            st.plotly_chart(prix_m2_static)
                            st.write("---")

                elif msg["step"]=="chart_histo_prix":

                    charts_prix_historique=get_chart_histo_prix()
                    with historique_prix_placeholder:                        
                        with st.container():                            
                            st.write(f"#### Evolution des prix sur 12 mois:", unsafe_allow_html=True)
                            # Afficher le graphique dans Streamlit
                            st.plotly_chart(charts_prix_historique)
                            st.write("---")
                elif msg["step"]=="chart_histo_activity":

                    charts_histo_activity=get_charts_histo_activity()
                    with historique_activity_placeholder:                        
                        with st.container():                            
                            st.write(f"#### Evolution des biens disponibles sur 12 mois:", unsafe_allow_html=True)
                            # Afficher le graphique dans Streamlit
                            st.plotly_chart(charts_histo_activity[0])
                            st.plotly_chart(charts_histo_activity[1])
                            st.write("---")

                elif msg["step"]=="recommandation_prix":
                    chunk=msg["value"]
                    # # Boucle pour afficher la réponse au fur et à mesure
                    with conseils_prix_placeholder:
                        trame_conseils_prix_full_response += chunk  # Ajoute chaque morceau de réponse
                        response_conseils_prix_placeholder.markdown(trame_conseils_prix_full_response, unsafe_allow_html=1)  


                        






    


    with st.expander("Voir l'analyse complète des images", expanded=0):
        st.markdown(st.session_state["resume_images"], unsafe_allow_html=1)