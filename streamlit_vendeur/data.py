import pandas as pd
from geopy.distance import geodesic
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import requests
import json
import geopandas as gpd
from shapely.geometry import Point, shape
import streamlit as st
import geojson
import folium
from pathlib import Path

# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent.resolve()
# Load environment variables
load_dotenv(dotenv_path="../.env", override=True)

@st.cache_data
def load_data():
    
    df=pd.read_csv(SCRIPT_DIR/"data/df_desciption_photos.csv")

    df.head(1).T
    cols=['typedebien', 'typedetransaction',
        'ville', 'nomQuartier','etage', 'surface','nb_pieces',
        'prix_bien','description_bien','nb_etages', 'parking',"annee_construction",
        "nb_logements_copro",  'prix_m_carre', 'chauffage_energie',
        'code_dep', 'resume_fr'
        ]
    df.dropna(subset=["ville", "prix_bien", "nb_pieces", "surface", "prix_m_carre"], inplace=True)
    df= df.reset_index(drop=True)

    df_extended=pd.read_csv(SCRIPT_DIR/"data/annonces_immo_ventes_details_lite.csv.zip", compression="zip", dtype={"codeinsee": str, "code_dep": str})
    df_extended["code_dep"]=df_extended["codeinsee"].str[:2]

    return df, df_extended

df, df_extended= load_data()

def calc_distance_km(point1, point2):

    # Calcul de la distance entre les deux points en metres
    distance=10000
    try:
        distance = geodesic(point1, point2).m
    except Exception as e:
        print(e)
        print(f"point 1: {point1}\npoint 2: {point2}")
        
    return distance

st.cache_data
def get_insee_stats(iris, coords_ref_point):
    def shorten_poi_label(l):
        # if l.find("École élémentaire"):
        #     return "École élémentaire"
        # elif l.find("Salles multisports"):
        #     return "Salles multisports"
        pattern="Lycée d’enseignement gén"
        if len(l) > len(pattern):
            return l[:len(pattern)]+"..."
        else:
            return l

    # root_path="/home/chougar/Documents/GitHub/immo_vis/immo_vis/data/"
    insee_subPath="data/insee/base_ic/"

    code_dep=iris[:2]
    INSEE_COM=iris[:5]
    cols_activite_residents={
        # "INSEE_COM": "INSEE_COM",
        # "DEP": "DEP",
        # "IRIS": "IRIS",
        "P19_POP1564":	"Population de 15 à 64 ans",
        # "P19_ACT1564":	"Actifs 15-64 ans",
        # "P19_CHOM1564":	"Chômeurs 15-64 ans",
        # "P19_ACT_SUP34":	"Actifs Enseignement sup de niveau bac + 3 ou 4",
        # "P19_ACT_SUP5":	"Actifs Enseignement sup de niveau bac + 5 ou plus",
        # "P19_RETR1564":	"Retraités Préretraités 15-64 ans",
        # "C19_ACT1564_CS1":	"Actifs 15-64 ans Agriculteurs exploitants",
        # "C19_ACT1564_CS2":	"Actifs 15-64 ans Artisans, Comm., Chefs entr.",
        # "C19_ACT1564_CS3":	"Actifs 15-64 ans Cadres, Prof. intel. sup.",
        # "C19_ACT1564_CS4":	"Actifs 15-64 ans Prof. intermédiaires",
        # "C19_ACT1564_CS5":	"Actifs 15-64 ans Employés",
        # "C19_ACT1564_CS6":	"Actifs 15-64 ans Ouvriers",
        "Part des actifs": "Taux d'actifs",
        'Taux de chômage': 'Taux de chômage',
        'Part des diplômés bac+3 et supérieur': 'Taux de diplômés bac+3 et supérieur',
        'Part des diplômés bac+5 et supérieur': 'Taux de diplômés bac+5 et supérieur',
        "Part des Agriculteurs": "Part des Agriculteurs",
        "Part des Artisans, Comm., Chefs entr.": "Part des Artisans, Comm., Chefs entr.",
        "Part des Cadres, Prof. intel. sup.": "Part des Cadres, Prof. intel. sup.",
        "Part des Prof. intermédiaires": "Part des Prof. intermédiaires",        
        "Part des Employés":"Part des Employés",
        "Part des Ouvriers":"Part des Ouvriers",

    }
    

    cols_revenus_disponibles={
        # "INSEE_COM": "INSEE_COM",
        # "DEP": "DEP",
        # "IRIS": "IRIS",
        "DISP_TP6019":	"Taux de pauvreté",
        "DISP_Q119":	"1er quartile (€)",
        "DISP_MED19":	"Médiane (€)",
        "DISP_Q319":	"3e quartile (€)",
        "DISP_EQ19":	"Écart inter-quartile rapporté à la médiane",
    }
    cols_evol_struc_pop={
        # "INSEE_COM": "INSEE_COM",
        # "DEP": "DEP",
        # "IRIS": "IRIS",
        "P19_POP":       "Population totale",
        # "P19_POP0014":	"Pop 0-14 ans",
        # "P19_POP1529":	"Pop 15-29 ans",
        # "P19_POP6074":	"Pop 60-74 ans",
        # "P19_POP75P":	"Pop 75 ans ou plus",
        "Part des moins de 30 ans": "Part des moins de 30 ans",
        "Part des plus de 60 ans": "Part des plus de 60 ans",
    }
    cols_couples_familles={
        # "INSEE_COM": "INSEE_COM",
        # "DEP": "DEP",
        # "IRIS": "IRIS",
        "C19_MEN":	"Ménages",
        # "C19_MENPSEUL":	"Ménages 1 personne",
        # "C19_MENFAM":	"Ménages avec famille(s)",
        # "C19_COUPAENF":	"Familles avec enfant(s)",
        # "C19_FAMMONO":	"Familles Monoparentales",
        # "C19_COUPSENF":	"Familles sans enfant",
        "Part des ménages avec 1 personne": "Part des ménages avec 1 personne",
        "Part des ménages avec famille(s)": "Part des ménages avec famille(s)",
        "Part des mamilles avec enfant(s)": "Part des mamilles avec enfant(s)",
        "Part des mamilles Monoparentales": "Part des mamilles Monoparentales",
        "Part des mamilles sans enfant": "Part des mamilles sans enfant",

    }

    activite_residents=pd.read_csv(SCRIPT_DIR/f"{insee_subPath}activite_residents_{code_dep}.csv", sep=";", dtype={"INSEE_COM":str,	"DEP": str, "IRIS":str})
    # traitements sur activité_residents
    activite_residents["Part des actifs"]=(activite_residents["P19_ACT1564"]/activite_residents["P19_POP1564"])*100
    activite_residents['Taux de chômage']=(activite_residents["P19_CHOM1564"]/activite_residents["P19_POP1564"])*100
    activite_residents['Part des diplômés bac+3 et supérieur']=np.round((activite_residents["P19_ACT_SUP34"]+activite_residents["P19_ACT_SUP5"])/activite_residents["P19_ACT1564"], 2)*100
    activite_residents['Part des diplômés bac+5 et supérieur']=np.round((activite_residents["P19_ACT_SUP5"])/activite_residents["P19_ACT1564"], 2)*100
    activite_residents["Part des Agriculteurs"]=(activite_residents["C19_ACT1564_CS1"]/activite_residents["P19_ACT1564"])*100
    activite_residents["Part des Artisans, Comm., Chefs entr."]=(activite_residents["C19_ACT1564_CS2"]/activite_residents["P19_ACT1564"])*100
    activite_residents["Part des Cadres, Prof. intel. sup."]=(activite_residents["C19_ACT1564_CS3"]/activite_residents["P19_ACT1564"])*100
    activite_residents["Part des Prof. intermédiaires"]=(activite_residents["C19_ACT1564_CS4"]/activite_residents["P19_ACT1564"])*100
    activite_residents["Part des Employés"]=(activite_residents["C19_ACT1564_CS5"]/activite_residents["P19_ACT1564"])*100
    activite_residents["Part des Ouvriers"]=(activite_residents["C19_ACT1564_CS6"]/activite_residents["P19_ACT1564"])*100


    revenus_disponibles=pd.read_csv(SCRIPT_DIR/f"{insee_subPath}revenus_disponibles_{code_dep}.csv", sep=";", dtype={"INSEE_COM":str,	"DEP": str, "IRIS":str})

    evol_struc_pop=pd.read_csv(SCRIPT_DIR/f"{insee_subPath}evol_struc_pop_{code_dep}.csv", sep=";", dtype={"INSEE_COM":str,	"DEP": str, "IRIS":str})
    # traitements population
    evol_struc_pop["Part des moins de 30 ans"]=((evol_struc_pop["P19_POP0014"]+evol_struc_pop["P19_POP1529"])/evol_struc_pop["P19_POP"])*100
    evol_struc_pop["Part des plus de 60 ans"]=((evol_struc_pop["P19_POP75P"]+evol_struc_pop["P19_POP1529"])/evol_struc_pop["P19_POP"])*100

    couples_familles_menages=pd.read_csv(SCRIPT_DIR/f"{insee_subPath}couples_familles_menages_{code_dep}.csv", sep=";", dtype={"INSEE_COM":str,	"DEP": str, "IRIS":str})
    # traitements sur couples_familles_menages
    couples_familles_menages["Part des ménages avec 1 personne"]=(couples_familles_menages["C19_MENPSEUL"]/couples_familles_menages["C19_MEN"])*100
    couples_familles_menages["Part des ménages avec famille(s)"]=(couples_familles_menages["C19_MENFAM"]/couples_familles_menages["C19_MEN"])*100
    couples_familles_menages["Part des mamilles avec enfant(s)"]=(couples_familles_menages["C19_COUPAENF"]/couples_familles_menages["C19_MEN"])*100
    couples_familles_menages["Part des mamilles Monoparentales"]=(couples_familles_menages["C19_FAMMONO"]/couples_familles_menages["C19_MEN"])*100
    couples_familles_menages["Part des mamilles sans enfant"]=(couples_familles_menages["C19_COUPSENF"]/couples_familles_menages["C19_MEN"])*100
    
    poi=pd.read_csv(SCRIPT_DIR/f"data/poi/poi_{code_dep}.csv", sep=";", dtype={"DEPCOM":str, "DEP": str, "DCIRIS":str})
    poi_filtre=[
        "A101",
        "A104",
        "A203",
        "A206",
        "A207",
        "A208",
        "A504",
        "B101",
        "B102",
        "B201",
        "B202",
        "B203",
        "B204",
        "B205",
        "B206",
        "B301",
        "C101",
        "C102",
        "C104",
        "C105",
        "C201",
        "C301",
        "C302",
        "D106",
        "D107",
        "D108",
        "D110",
        "D112",
        "D113",
        "D201",
        "D232",
        "D233",
        "D301",
        "D502",
        "F101",
        "F102",
        "F103",
        "F104",
        "F105",
        "F106",
        "F107",
        "F108",
        "F109",
        "F111",
        "F112",
        "F113",
        "F114",
        "F116",
        "F117",
        "F118",
        "F119",
        "F120",
        "F121",
        "F201",
        "F203",
        "F303",
        "F304",
        "F305",
        "F306",
        "F307",
        "G102",
    ]
    poi=poi[(poi["DEPCOM"]==INSEE_COM)&(~pd.isna(poi["lat"]))&(poi["TYPEQU"].isin(pd.Series(poi_filtre)))]
    # traitements sur les poi
    poi['distance_m'] = poi.apply(lambda row: np.round(calc_distance_km(coords_ref_point, (row['lng'], row['lat'])), 1), axis=1)
    poi=poi[poi['distance_m']<1500]
    ref_poi=pd.read_csv(SCRIPT_DIR/f"data/ref_type_equip.csv", sep=";")
    poi=pd.merge(left=poi, right=ref_poi, left_on="TYPEQU", right_on="type", how="inner")

    cols_to_drop=['INSEE_COM', "DEP", "IRIS"]       
    couples_familles_menages_iris= couples_familles_menages[couples_familles_menages["IRIS"]==iris].drop(columns=cols_to_drop)
    couples_familles_menages_iris= couples_familles_menages_iris.round(0).sample(1).rename(columns=cols_couples_familles)[cols_couples_familles.values()].reset_index(drop=True).to_dict('records')[0]
    activite_residents_iris= activite_residents[activite_residents["IRIS"]==iris].round(0).sample(1).rename(columns=cols_activite_residents)[cols_activite_residents.values()].reset_index(drop=True).to_dict('records')[0]
    revenus_disponibles_iris= revenus_disponibles[revenus_disponibles["IRIS"]==iris].round(0).sample(1).rename(columns=cols_revenus_disponibles)[cols_revenus_disponibles.values()].reset_index(drop=True).to_dict('records')[0]
    evol_struc_pop_iris= evol_struc_pop[evol_struc_pop["IRIS"]==iris].round(0).sample(1).rename(columns=cols_evol_struc_pop)[cols_evol_struc_pop.values()].reset_index(drop=True).to_dict('records')[0]

    
    activite_residents_com= activite_residents[activite_residents["INSEE_COM"]==INSEE_COM].round(0).reset_index(drop=True).round(1)
    evol_struc_pop_com= evol_struc_pop[evol_struc_pop["INSEE_COM"]==INSEE_COM].round(0).reset_index(drop=True).round(1)
    revenus_disponibles_com= revenus_disponibles[revenus_disponibles["INSEE_COM"]==INSEE_COM].round(0).reset_index(drop=True).round(1)

    poi_focus=poi.groupby('libelle_equipement').agg({"libelle_equipement": "count"}).rename(columns={"libelle_equipement": "Nombre de POI"}).sort_values(by="Nombre de POI", ascending=False).head(15)
    
    poi["libelle_equipement"]=poi["libelle_equipement"].apply(shorten_poi_label)

    return {       
            "Données pour le quartier":{
                "Couples_familles_menages": couples_familles_menages_iris,
                "Situation des actifs":activite_residents_iris,
                "Revenus et pauvreté":revenus_disponibles_iris,
                "Démographie": evol_struc_pop_iris
            },
            "Données pour la commune": {
                "Situation des actifs": {
                    'Population de 15 à 64 ans': activite_residents_com["P19_POP1564"].sum(),
                    "Part des actifs": (activite_residents_com["P19_ACT1564"].sum()/ activite_residents_com["P19_POP1564"].sum())*100,
                    'Taux de chômage': (activite_residents_com["P19_CHOM1564"].sum()/ activite_residents_com["P19_POP1564"].sum())*100,
                    'Part des diplômés bac+3 et supérieur':np.round((activite_residents_com["P19_ACT_SUP34"].sum()+activite_residents_com["P19_ACT_SUP5"].sum())/activite_residents_com["P19_ACT1564"].sum(), 2)*100,
                    'Part des diplômés bac+5 et supérieur':np.round((activite_residents_com["P19_ACT_SUP5"].sum())/activite_residents_com["P19_ACT1564"].sum(), 2)*100,
                    "Part des Agriculteurs":np.round(activite_residents_com["C19_ACT1564_CS1"].sum()/activite_residents_com["P19_ACT1564"].sum(), 2)*100,
                    "Part des Artisans, Comm., Chefs entr.": np.round(activite_residents_com["C19_ACT1564_CS2"].sum()/activite_residents_com["P19_ACT1564"].sum(), 2)*100,
                    "Part des Cadres, Prof. intel. sup.":np.round(activite_residents_com["C19_ACT1564_CS3"].sum()/activite_residents_com["P19_ACT1564"].sum(), 2)*100,
                    "Part des Prof. intermédiaires":np.round(activite_residents_com["C19_ACT1564_CS4"].sum()/activite_residents_com["P19_ACT1564"].sum(), 2)*100,
                    "Part des Employés":np.round(activite_residents_com["C19_ACT1564_CS5"].sum()/activite_residents["P19_ACT1564"].sum(), 2)*100,
                    "Part des Ouvriers" :np.round(activite_residents_com["C19_ACT1564_CS6"].sum()/activite_residents["P19_ACT1564"].sum(), 2)*100
                },
                "Démographie": {
                    "Population": evol_struc_pop_com["P19_POP"].sum(),
                    "Part des moins de 30 ans": np.round((evol_struc_pop_com["P19_POP0014"].sum()+evol_struc_pop_com["P19_POP1529"].sum())/evol_struc_pop_com["P19_POP"].sum(), 1)*100,
                    "Part des plus de 60 ans": np.round((evol_struc_pop_com["P19_POP75P"].sum()+evol_struc_pop_com["P19_POP1529"].sum())/evol_struc_pop_com["P19_POP"].sum(),1)*100,
                },
                "Revenus et pauvreté": {
                    "Taux de pauvreté": np.round(revenus_disponibles_com["DISP_TP6019"].mean(), 2),                    
                    "Médiane (€)": np.round(revenus_disponibles_com["DISP_MED19"].mean(), 2),
                }
            },
            "Commerces et services à moins de 1.5 km": poi_focus.to_dict(),
            "sample_poi": poi.sample(100)
    }



def get_locals(adresse):
    
    query=f"""https://api-adresse.data.gouv.fr/search/?q={adresse.replace(" ", "+")}&limit=5"""

    resp=requests.get(query)
    if resp.status_code==200:
        resp=resp.json()
                
        coord=resp["features"][0]["geometry"]["coordinates"]#["citycode"]
        dep_adresse=resp["features"][0]["properties"]["citycode"][:2]
        lieu=resp["features"][0]["properties"]["city"]+", "+resp["features"][0]["properties"]["context"]
        
        iris_data= get_iris(coord, dep_adresse)

        return {"iris": iris_data[0], "lib_quartier": iris_data[1], "latLon": coord, "lieu_label": lieu}
        
    else:
        print(resp.status_code)
        return None

st.cache_data
def get_iris(coord, dep_adresse):

    point = Point(coord)

    # iris_poly_path="/home/chougar/Documents/GitHub/immo_vis/immo_vis/data/ref/polygons/"
    # with open(f"{iris_poly_path}polygones_{dep_adresse}.json") as f:
    with open(SCRIPT_DIR/f"data/polygons/polygones_{dep_adresse}.json") as f:
        iris_polygons = json.load(f)

    ref_iris=pd.read_csv(SCRIPT_DIR/"data/df_reference_communes_iris.csv", sep=";", dtype={"CODE_IRIS": str})

    # Collect all the features
    features = []
    for key in iris_polygons:
        feature = iris_polygons[key]['polygone']
        features.append(feature)

    # Create a FeatureCollection
    feature_collection = geojson.FeatureCollection(features)

    # Create the Point (longitude, latitude)
    point = Point(coord)

    # Iterate through the features and check containment
    for feat in feature_collection['features']:
        try:
            polygon = shape(feat['geometry'])
            if polygon.contains(point):
                code_iris = feat['properties']['CODE_IRIS']
                #print(f"The coordinate lies within CODE_IRIS: {code_iris}")
                
                quartier=""
                try:
                    quartier=ref_iris[ref_iris["CODE_IRIS"]==code_iris]["LIB_IRIS"].values[0]
                except Exception as e:
                    quartier="Libelle indisponible"
                return code_iris, quartier

        except Exception as e:
            print(f"Error processing feature {feat['id']}: {e}")
    else:
        print("The coordinate does not lie within any of the polygons.")    



def get_price_stats(iris, coords_ref_point, nb_pieces, surface, type_bien, type_transaction):    
    com=iris[:5]
    dep=iris[:2]


    df_filtre_com=df_extended[(df_extended["codeinsee"]==com)&
                          (df_extended['typedebien']==type_bien)&
                          (df_extended["typedetransaction"]==type_transaction)&
                          (df_extended["nb_pieces"]==nb_pieces)&
                          (df_extended["surface"].between(surface-10, surface+100))]
    prix_median_commune=df_filtre_com["prix_bien"].median()

    df_filtre_com["distance_m"]= df_filtre_com.apply(lambda row: np.round(calc_distance_km(coords_ref_point, (row['mapCoordonneesLongitude'], row['mapCoordonneesLatitude'])), 1), axis=1)    
    df_filtre_distance=df_filtre_com[(df_filtre_com["distance_m"]<2000)&
                          (df_filtre_com['typedebien']==type_bien)&
                          (df_filtre_com["typedetransaction"]==type_transaction)&
                          (df_filtre_com["nb_pieces"]==nb_pieces)]
    prix_median_rayon_2000=df_filtre_distance["prix_bien"].median()

    df_filtre_dep=df_extended[(df_extended["code_dep"]==dep)&
                (df_extended['typedebien']==type_bien)&
                (df_extended["typedetransaction"]==type_transaction)&
                (df_extended["nb_pieces"]==nb_pieces)&
                (df_extended["surface"].between(surface-10, surface+10))]
    prix_median_dep=df_filtre_dep["prix_bien"].median()

    data=[
        {"label": "Prix médian à 2000m", "value": prix_median_rayon_2000},
        {"label": "Prix médian commune", "value": prix_median_commune},
        {"label": "Prix médian département", "value": prix_median_dep}
    ]

    return {
        "text":f"""
        Relevé des prix autour du bien:
            Prix médian sur un rayon de 2000m: {prix_median_rayon_2000} (basé sur {len(df_filtre_distance)} annonces),\n 
            Prix médian dans la commune: {prix_median_commune} (basé sur {len(df_filtre_com)} annonces),\n 
            Prix médian dans le département: {prix_median_dep} (basé sur {(len(df_filtre_dep))} annonces)
            """,
        "data": data
    }

def get_resume_neutral(r):
    # sep pour resume pices
    seps=["### conclusion générale", "### conclusion finale"]
    resume_pieces=""
    for sep in seps:
        if sep in r.lower():
            resume_pieces=r.lower().split(sep)
            if len(resume_pieces)>0:
                resume_pieces=resume_pieces[0]
                break
  

    seps=["### points clés", "**points clés"]
    desc_finale=""
    for sep in seps:
        if sep in r.lower():
            desc_finale=r.lower().split(sep)
            if len(desc_finale)>0:
                desc_finale= sep+desc_finale[-1]
            resume_complet=resume_pieces+"\n\n"+desc_finale
            break
    return resume_complet


client = OpenAI()


def build_photos_album(idannonce):
    import os
    from PIL import Image
    import matplotlib.pyplot as plt

    source_dir = "/home/chougar/Documents/GitHub/image-to-text-immo/photos/"
    for folder_name in os.listdir(source_dir):
        if folder_name.startswith(idannonce):
            print(folder_name)


            # Liste des fichiers images (supposons qu'ils sont au format JPG ou PNG)
            image_files = [os.path.join(source_dir+folder_name, f) for f in os.listdir(source_dir+folder_name) if f.endswith(('.jpg', '.png'))][:9]
            print(image_files)


            # Créer une figure Matplotlib de 3x3
            fig, axes = plt.subplots(3, 3, figsize=(10, 10))  # Ajustez la taille de la figure

            # Afficher chaque image dans une sous-figure
            for i, ax in enumerate(axes.flat):
                if i < len(image_files):
                    img = Image.open(image_files[i])
                    ax.imshow(img)
                    ax.axis('off')  # Désactiver les axes
                    ax.set_facecolor('black')
                else:
                    ax.axis('off')  # Masquer les sous-figures vides
                    ax.set_facecolor('black')

            # Ajuster l'espacement entre les images
            plt.tight_layout()


            # Sauvegarder la galerie
            fig.savefig("galerie_3x3.png", bbox_inches='tight', pad_inches=0.1, facecolor='black')

            return
def build_ad_description_input(adresse_du_bien, type_bien, type_transaction, nb_pieces, surface, prix=None):
    locals=get_locals(adresse_du_bien)

    (iris, lib_quartier, lieu_label, latLng, insee_stats)=(None, "", None, None, "")
    if locals and len(locals)==4:
        iris=locals["iris"]
        lib_quartier=locals["lib_quartier"]
        latLng=locals["latLon"]
        lieu_label=locals["lieu_label"]

    if iris:
        insee_stats= get_insee_stats(iris, latLng)
        data_poi=insee_stats["sample_poi"]
        del insee_stats["sample_poi"]

        
    
    pool_biens=df[(df["typedebien"]==type_bien)&(df["nb_pieces"]>=nb_pieces)&(df["surface"]>=surface)].reset_index(drop=True)
    random_bien=pool_biens.sample(1).reset_index(drop=True)

    idannonce=random_bien['idannonce'].values[0]
    
    print(f"idannonce: {idannonce}")
    build_photos_album(idannonce)

    resume_fr=get_resume_neutral(random_bien.loc[0, :][["resume_fr"]].values[0])
    
    releve_prix=get_price_stats(iris, latLng, nb_pieces, surface, type_bien, type_transaction)
    data_prix=releve_prix["data"]
    data_prix.append({"label": "Prix demandé", "value": prix})
    data_prix=pd.DataFrame(data_prix).fillna(0)

    releve_prix=releve_prix["text"]
      

    prompt=f"""
        1. Fiche du bien:\n
        Type de bien: {type_bien},\n
        Type de transaction: {type_transaction},\n
        Commune: {lieu_label},\n
        Quartier: {lib_quartier},\n
        Nombre de pièces: {nb_pieces},\n
        Surface: {surface},\n
        Prix: {prix},\n
        Prix au m2: {np.round((prix/surface), 0)}\n
        \n
        2. Description du bien à partir des photos (traitement image-to-text):\n        
        {resume_fr}
        \n
        3. Données chiffrées sur le quartier et la commune du bien:\n
        {insee_stats}\n
        4. Relevé de prix des biens similaires: {releve_prix}
    """    

    return prompt, data_poi, data_prix, latLng
    


def buimd_map_poi(data_poi, coords_ref_point):
    _df=data_poi
    # Calculer le centre de la carte
    map_center = [_df['lat'].mean(), _df['lng'].mean()]

    # Créer une carte Folium
    m = folium.Map(location=map_center, zoom_start=15,)# tiles='cartodbpositron')


    # Ajouter les marqueurs pour chaque POI
    for index, row in _df.iterrows():
        folium.Marker(
            location=[row['lat'], row['lng']],
            tooltip=row['libelle_equipement'],
            icon=folium.Icon(color=row['colorMarker'], icon_color=row['colorIcon'],)# icon=row["category_iconMarker"])
        ).add_to(m)


    # Ajouter un marqueur personnalisé avec une icône "maison" et une taille de 150%
    folium.Marker(
        location=(coords_ref_point[1], coords_ref_point[0]),
        tooltip="Mon bien",
        icon=folium.DivIcon(
            icon_size=(30, 30),  # Taille de l'icône (largeur, hauteur)
            icon_anchor=(15, 15),  # Point d'ancrage de l'icône (centre)
            html='<div style="font-size: 400%; color: blue;"><i class="fa fa-home"></i></div>'
        )
    ).add_to(m)    



    # Générer la légende en HTML
    legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: auto; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    opacity: 0.85;
                    padding: 10px;">
            <p><strong>Légende</strong></p>
    '''

    # Ajouter chaque couleur et libellé à la légende
    for color, label in zip(_df['colorHEX'].unique(), _df['libelle_equipement'].unique()):
        legend_html += f'''
            <p><i class="fa fa-circle" style="color:{color};"></i> {label}</p>
        '''

    legend_html += '</div>'
    
    return m, legend_html


metastore_plots={"data_poi":[], "data_price": []}

def generate_description(adresse_du_bien, type_bien, type_transaction, nb_pieces, surface, prix=None):
    # adresse_du_bien="57 quai georges gorce, boulogne bill"
    # # type_bien_vendeur="Maison/Villa"
    # type_bien_vendeur="Appartement"
    # nb_pieces=4
    # surface=70
    
    if prix==None:
        prix=np.random.randint(300000, 600000)
    
    data_descr=build_ad_description_input(adresse_du_bien, type_bien, type_transaction, nb_pieces, surface, prix)
    prompt_user=data_descr[0]
    metastore_plots["data_poi"]=data_descr[1]
    metastore_plots["data_prix"]=data_descr[2]
    metastore_plots["coords_ref"]=data_descr[3]
    

    
    messages=[
            {"role": "system",  "content": (
                "Vous êtes un expert en rédaction d'annonces immobilières. "
                "Votre tâche est de créer une description détaillée et structurée pour un bien immobilier à vendre ou à louer, "
                "à partir des informations fournies. Adoptez un style professionnel et engageant, tout en respectant la structure suivante :\n"
                "\n"
                "1. **Description de l'appartement** : Une description concise et valorisante du bien (environ 100 à 150 mots), "
                "incluant ses caractéristiques principales, son état général, et son attrait global.\n"
                "2. **Économie & Démographie** : Un aperçu synthétique du quartier et de la commune, mettant en avant les statistiques utiles "
                "pour situer le niveau de vie et l'environnement socio-économique du quartier par rapport à la commune. \n"
                "   - **Population et ménages** : Donnez des chiffres pertinents sur les habitants et la structure des ménages, "
                "en comparant les données du quartier à celles de la commune.\n"
                "   - **Revenus et emploi** : Présentez les revenus et le marché de l’emploi (chômage, professions principales) avec des comparaisons utiles.\n"
                "3. **Commerces & services** : Dressez une liste des commerces et services proches du bien, en soulignant les éléments "
                "qui améliorent la qualité de vie (restauration, santé, éducation, sports, etc.).\n"
                "4. Relevé de prix des biens similaires: compare le prix demandé par le vendeur au relevé des prix fourni, et conseille un ajustement si nécessaire"
            )},
            {"role": "user", "content": prompt_user}
    ]
    # image to text gpt vs llama

    

    client = OpenAI()


    response_gpt = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
     temperature=0.7,
        stream=True  # Active le streaming
    )

    # Itération sur les morceaux de réponse
    for chunk in response_gpt:
        if chunk.choices[0].delta.content:  # Vérifie si le morceau contient du texte            
            yield chunk.choices[0].delta.content  # Renvoie le morceau de texte
    return prompt_user

   
def get_plots():
    _map=buimd_map_poi(metastore_plots["data_poi"], metastore_plots["coords_ref"])
    legend=_map[1]
    _map=_map[0]
    
    return _map, legend, metastore_plots["data_prix"]



