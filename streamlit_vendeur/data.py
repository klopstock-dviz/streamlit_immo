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
import shutil
import os
from PIL import Image
import plotly.express as px


# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent.resolve()
# Load environment variables
load_dotenv(dotenv_path="../.env", override=True)
api_key=os.getenv("OPENAI_API_KEY_streamlit_immo")

@st.cache_data
def load_data(code_dep, transaction_type):
    
    df_photos_desc=pd.read_csv(SCRIPT_DIR/"data/df_desciption_photos.csv")

    df_photos_desc.head(1).T
    cols=['typedebien', 'typedetransaction',
        'ville', 'nomQuartier','etage', 'surface','nb_pieces',
        'prix_bien','description_bien','nb_etages', 'parking',"annee_construction",
        "nb_logements_copro",  'prix_m_carre', 'chauffage_energie',
        'code_dep', 'resume_fr'
        ]
    df_photos_desc.dropna(subset=["ville", "prix_bien", "nb_pieces", "surface", "prix_m_carre"], inplace=True)
    df_photos_desc= df_photos_desc.reset_index(drop=True)

    # df_extended=pd.read_csv(SCRIPT_DIR/"data/annonces_immo_ventes_details_lite.csv.zip", compression="zip", dtype={"codeinsee": str, "code_dep": str})

    df_immo = pd.read_csv(
        f"https://raw.githubusercontent.com/klopstock-dviz/immo_vis/refs/heads/master/data/annonces_git/df_annonces_gps_iris_{transaction_type.lower()}s_{code_dep}.csv", 
        sep=";",  dtype={"INSEE_COM": str})

    df_immo["code_dep"]=df_immo["INSEE_COM"].str[:2]

    return df_photos_desc, df_immo



def calc_distance_km(point1, point2):

    # Calcul de la distance entre les deux points en metres
    distance=10000
    try:
        distance = geodesic(point1, point2).m
    except Exception as e:
        print(e)
        print(f"point 1: {point1}\npoint 2: {point2}")
        
    return distance

@st.cache_data
def get_insee_stats(iris: str, coords_ref_point: list, is_iris: bool):
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
    insee_subPath="https://raw.githubusercontent.com/klopstock-dviz/immo_vis/refs/heads/master/data/insee/base_ic/"

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

    activite_residents=pd.read_csv(f"{insee_subPath}activite_residents_{code_dep}.csv", sep=";", dtype={"INSEE_COM":str,	"DEP": str, "IRIS":str})
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


    revenus_disponibles=pd.read_csv(f"{insee_subPath}revenus_disponibles_{code_dep}.csv", sep=";", dtype={"INSEE_COM":str,	"DEP": str, "IRIS":str})

    evol_struc_pop=pd.read_csv(f"{insee_subPath}evol_struc_pop_{code_dep}.csv", sep=";", dtype={"INSEE_COM":str,	"DEP": str, "IRIS":str})
    # traitements population
    evol_struc_pop["Part des moins de 30 ans"]=((evol_struc_pop["P19_POP0014"]+evol_struc_pop["P19_POP1529"])/evol_struc_pop["P19_POP"])*100
    evol_struc_pop["Part des plus de 60 ans"]=((evol_struc_pop["P19_POP75P"]+evol_struc_pop["P19_POP1529"])/evol_struc_pop["P19_POP"])*100

    couples_familles_menages=pd.read_csv(f"{insee_subPath}couples_familles_menages_{code_dep}.csv", sep=";", dtype={"INSEE_COM":str,	"DEP": str, "IRIS":str})
    # traitements sur couples_familles_menages
    couples_familles_menages["Part des ménages avec 1 personne"]=(couples_familles_menages["C19_MENPSEUL"]/couples_familles_menages["C19_MEN"])*100
    couples_familles_menages["Part des ménages avec famille(s)"]=(couples_familles_menages["C19_MENFAM"]/couples_familles_menages["C19_MEN"])*100
    couples_familles_menages["Part des mamilles avec enfant(s)"]=(couples_familles_menages["C19_COUPAENF"]/couples_familles_menages["C19_MEN"])*100
    couples_familles_menages["Part des mamilles Monoparentales"]=(couples_familles_menages["C19_FAMMONO"]/couples_familles_menages["C19_MEN"])*100
    couples_familles_menages["Part des mamilles sans enfant"]=(couples_familles_menages["C19_COUPSENF"]/couples_familles_menages["C19_MEN"])*100
    

    # poi=pd.read_csv(SCRIPT_DIR/f"data/poi/poi_{code_dep}.csv", sep=";", dtype={"DEPCOM":str, "DEP": str, "DCIRIS":str})
    poi_adress="https://raw.githubusercontent.com/klopstock-dviz/immo_vis/refs/heads/master/data/poi/"
    poi=pd.read_csv(f"{poi_adress}poi_{code_dep}.csv", sep=";", dtype={"DEPCOM":str, "DEP": str, "DCIRIS":str})
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
    poi=poi[poi['distance_m']<1000]
    ref_poi=pd.read_csv(
        "https://raw.githubusercontent.com/klopstock-dviz/immo_vis/refs/heads/master/data/ref/ref_type_equip.csv", 
        sep=";"
    )
    
    poi=pd.merge(left=poi, right=ref_poi, left_on="TYPEQU", right_on="type", how="inner")

    cols_to_drop=['INSEE_COM', "DEP", "IRIS"]       
    couples_familles_menages_iris= couples_familles_menages[couples_familles_menages["IRIS"]==iris].drop(columns=cols_to_drop)
    couples_familles_menages_iris= couples_familles_menages_iris.round(0).sample(1).rename(columns=cols_couples_familles)[cols_couples_familles.values()].reset_index(drop=True).to_dict('records')[0]
    activite_residents_iris= activite_residents[activite_residents["IRIS"]==iris]
    
    revenus_disponibles_iris= revenus_disponibles[revenus_disponibles["IRIS"]==iris]

    # gestion cas commune sans iris pour tx pauvreté et revenus (fichier filosi)
    revenus_disponibles_com, revenus_disponibles_iris=(None, None)
    if is_iris:
        revenus_disponibles_iris= revenus_disponibles[revenus_disponibles["IRIS"]==iris].round(0).sample(1).rename(columns=cols_revenus_disponibles)[cols_revenus_disponibles.values()].reset_index(drop=True).to_dict('records')[0]
    
    # revenu pour la commune issu de la base_cc comparateur communes
    df_base_cc=  pd.read_csv(
        f"https://raw.githubusercontent.com/klopstock-dviz/immo_vis/refs/heads/master/data/insee/communes/base_cc_comparateur/base_cc_comparateur_{code_dep}.csv", 
        sep=";",
        dtype={"INSEE_COM": str}
    )
    revenus_disponibles_com=df_base_cc[df_base_cc["INSEE_COM"]==INSEE_COM]
    cols={"MED19": "Médiane de niveau de vie (€)", "TP6019": 'Taux de pauvreté'}
    revenus_disponibles_com= revenus_disponibles_com.round(0).sample(1).rename(columns=cols)[cols.values()].reset_index(drop=True).to_dict('records')[0]

    
    evol_struc_pop_iris= evol_struc_pop[evol_struc_pop["IRIS"]==iris].round(0).sample(1).rename(columns=cols_evol_struc_pop)[cols_evol_struc_pop.values()].reset_index(drop=True).to_dict('records')[0]

    
    activite_residents_com= activite_residents[activite_residents["INSEE_COM"]==INSEE_COM].round(0).reset_index(drop=True).round(1)
    evol_struc_pop_com= evol_struc_pop[evol_struc_pop["INSEE_COM"]==INSEE_COM].round(0).reset_index(drop=True).round(1)
    

    poi_focus=poi.groupby('libelle_equipement').agg({"libelle_equipement": "count"}).rename(columns={"libelle_equipement": "Nombre de POI"}).sort_values(by="Nombre de POI", ascending=False).head(15)
    
    poi.loc[:, "libelle_equipement"]=poi["libelle_equipement"].apply(shorten_poi_label)

    def part_metiers_niv_geo(_activite_residents):
        metiers_stats_locales=[
            {"cat": "Part des Agriculteurs", "value": float(np.round(_activite_residents["C19_ACT1564_CS1"].sum()/_activite_residents["P19_ACT1564"].sum(), 2))*100},
            {"cat": "Part des Artisans, Comm., Chefs entr.", "value":  float(np.round(_activite_residents["C19_ACT1564_CS2"].sum()/_activite_residents["P19_ACT1564"].sum(), 2))*100},
            {"cat": "Part des Cadres, Prof. intel. sup.", "value":  float(np.round(_activite_residents["C19_ACT1564_CS3"].sum()/_activite_residents["P19_ACT1564"].sum(), 2))*100},
            {"cat": "Part des Prof. intermédiaires", "value":  float(np.round(_activite_residents["C19_ACT1564_CS4"].sum()/_activite_residents["P19_ACT1564"].sum(), 2))*100},
            {"cat": "Part des Employés", "value":  float(np.round(_activite_residents["C19_ACT1564_CS5"].sum()/_activite_residents["P19_ACT1564"].sum(), 2))*100},
            {"cat": "Part des Ouvriers", "value":  float(np.round(_activite_residents["C19_ACT1564_CS6"].sum()/_activite_residents["P19_ACT1564"].sum(), 2))*100}
        ]    

        df_metiers_stats_locales= pd.DataFrame(metiers_stats_locales).sort_values(by="value", ascending=0).head(2)

        return df_metiers_stats_locales.set_index('cat')['value'].to_dict()


    if is_iris:
        top_metiers_iris=part_metiers_niv_geo(activite_residents_iris)
        stats_quartier={
            "Couples_familles_menages": couples_familles_menages_iris,
            "Situation des actifs":top_metiers_iris,
            'Population de 15 à 64 ans': float(activite_residents_iris["P19_POP1564"].sum()),
            "Part des actifs": float((activite_residents_iris["P19_ACT1564"].sum()/ activite_residents_iris["P19_POP1564"].sum()))*100,
            'Taux de chômage': float(activite_residents_iris["P19_CHOM1564"].sum()/ activite_residents_iris["P19_POP1564"].sum())*100,
            'Part des diplômés bac+3 et supérieur':float(np.round((activite_residents_iris["P19_ACT_SUP34"].sum()+activite_residents_iris["P19_ACT_SUP5"].sum())/activite_residents_iris["P19_ACT1564"].sum(), 2))*100,
            'Part des diplômés bac+5 et supérieur':float(np.round((activite_residents_iris["P19_ACT_SUP5"].sum())/activite_residents_iris["P19_ACT1564"].sum(), 2))*100,
            "Revenus et pauvreté":revenus_disponibles_iris,
            "Démographie": evol_struc_pop_iris
        }
    else:
        stats_quartier="Absentes"                
    
    top_metiers_commune=part_metiers_niv_geo(activite_residents_com)

    top_metiers_dep=part_metiers_niv_geo(activite_residents)

    return {
        "stats_for_llm":{
            "Données pour le quartier": stats_quartier,
            "Données pour la commune": {
                "Situation des actifs": {
                    'Population de 15 à 64 ans': float(activite_residents_com["P19_POP1564"].sum()),
                    "Part des actifs": float(activite_residents_com["P19_ACT1564"].sum()/ activite_residents_com["P19_POP1564"].sum())*100,
                    'Taux de chômage': float(activite_residents_com["P19_CHOM1564"].sum()/ activite_residents_com["P19_POP1564"].sum())*100,
                    'Part des diplômés bac+3 et supérieur':float(np.round((activite_residents_com["P19_ACT_SUP34"].sum()+activite_residents_com["P19_ACT_SUP5"].sum())/activite_residents_com["P19_ACT1564"].sum(), 2))*100,
                    'Part des diplômés bac+5 et supérieur':float(np.round((activite_residents_com["P19_ACT_SUP5"].sum())/activite_residents_com["P19_ACT1564"].sum(), 2))*100,
                    "Top 2 des métiers les plus représentés (%)":top_metiers_commune
                },
                "Démographie": {
                    "Population": float(evol_struc_pop_com["P19_POP"].sum()),
                    "Part des moins de 30 ans": float(np.round((evol_struc_pop_com["P19_POP0014"].sum()+evol_struc_pop_com["P19_POP1529"].sum())/evol_struc_pop_com["P19_POP"].sum(), 1))*100,
                    "Part des plus de 60 ans": float(np.round((evol_struc_pop_com["P19_POP75P"].sum()+evol_struc_pop_com["P19_POP1529"].sum())/evol_struc_pop_com["P19_POP"].sum(),1))*100,
                },
                "Revenus et pauvreté": revenus_disponibles_com
            },            
            "Données pour le département": {
                "Situation des actifs": {
                    "Part des actifs": float(np.round((activite_residents["P19_ACT1564"].sum() / activite_residents["P19_POP1564"].sum()) * 100, 2)),
                    'Taux de chômage': float(np.round((activite_residents["P19_CHOM1564"].sum() / activite_residents["P19_POP1564"].sum()) * 100, 2)),
                    'Part des diplômés bac+3 et supérieur':float(np.round((activite_residents["P19_ACT_SUP34"].sum()+activite_residents["P19_ACT_SUP5"].sum())/activite_residents["P19_ACT1564"].sum(), 2)*100),
                    'Part des diplômés bac+5 et supérieur':float(np.round((activite_residents["P19_ACT_SUP5"].sum())/activite_residents["P19_ACT1564"].sum(), 2)*100),
                    "Top 2 des métiers les plus représentés (%)":top_metiers_dep
                },
                "Démographie": {
                    "Population": float(evol_struc_pop["P19_POP"].sum()),
                    "Part des moins de 30 ans": float(np.round((evol_struc_pop["P19_POP0014"].sum()+evol_struc_pop["P19_POP1529"].sum())/evol_struc_pop["P19_POP"].sum(), 1)*100),
                    "Part des plus de 60 ans": float(np.round((evol_struc_pop["P19_POP75P"].sum()+evol_struc_pop["P19_POP1529"].sum())/evol_struc_pop["P19_POP"].sum(),1)*100),
                },
                "Revenus et pauvreté": df_base_cc[["MED19", "TP6019"]].rename(columns=cols).median().to_dict()
            },
            "Commerces et services à moins de 1 km": poi_focus.to_dict(),        
        },
        "stats_for_map": poi.sample(min(len(poi), 100))
    }



def get_locals(adresse) -> dict:
    """
        input: adresse (str)<br>
        output: {"status_code": 200, "iris": str, "lib_quartier": str, "latLon": array, "lieu_label": str, "code_dep": str}
    """
    query=f"""https://api-adresse.data.gouv.fr/search/?q={adresse.replace(" ", "+")}&limit=5"""

    resp=requests.get(query)
    if resp.status_code==200:
        resp=resp.json()
                
        coord=resp["features"][0]["geometry"]["coordinates"]#["citycode"]
        code_dep=resp["features"][0]["properties"]["citycode"][:2]
        lieu=resp["features"][0]["properties"]["city"]+", "+resp["features"][0]["properties"]["context"]
        
        iris_data= get_iris(coord, code_dep)

        return {
            "status_code": 200, 
            "iris": iris_data[0], "lib_quartier": iris_data[1], "is_iris": iris_data[2] , 
            "latLon": coord, "lieu_label": lieu, "code_dep": code_dep
        }
        
    else:
        print(resp.status_code)
        return {"status_code": resp.status_code}

st.cache_data
def get_iris(coord: list, code_dep: str) -> dict:
    """
        inputs:
            * coords: lat & lng (array)
            * code_dep (str)
        output:
            * code_iris (str)
            * quartier (str)
    """

    point = Point(coord)


    iris_polygons=pd.read_json(f"https://raw.githubusercontent.com/klopstock-dviz/immo_vis/refs/heads/master/data/ref/polygons/polygones_{code_dep}.json")
    iris_polygons=json.loads(iris_polygons.to_json())

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
                code_insee = feat['properties']['INSEE_COM']
                is_iris=False if len(ref_iris[ref_iris["INSEE_COM"]==code_insee])==1 else True
                
                quartier=""
                try:
                    quartier=ref_iris[ref_iris["CODE_IRIS"]==code_iris]["LIB_IRIS"].values[0]
                except Exception as e:
                    quartier="Libelle indisponible"
                return code_iris, quartier, is_iris

        except Exception as e:
            print(f"Error processing feature {feat['id']}: {e}")
    else:
        print("The coordinate does not lie within any of the polygons.")    



def get_price_stats(df_immo, iris, coords_ref_point, nb_pieces, surface, type_bien, type_transaction):    
    com=iris[:5]
    dep=iris[:2]

    codes_maps={"Vente": "v", "Location": "l", "Appartement": "a", "Maison/Villa": "m"}

    df_immo["prix_m2"]=round(df_immo["prix_bien"]/df_immo["surface"],2)

    # normalisation des filtres
    margins_surface=10 if codes_maps[type_bien]=='a' else 20
    df_immo.loc[:, "date_norm"]=pd.to_datetime(df_immo["date"])
    newest_date=df_immo["date_norm"].max()
    oldest_date_short=newest_date-pd.DateOffset(months=4)
    oldest_date_medium=newest_date-pd.DateOffset(months=6)
    oldest_date_long=newest_date-pd.DateOffset(months=12)


    # commune
    df_filtre_com=df_immo[(df_immo["INSEE_COM"]==com)&
                          (df_immo['typedebien']==codes_maps[type_bien])&
                          (df_immo["typedetransaction"]==codes_maps[type_transaction])&
                          (df_immo["nb_pieces"]==nb_pieces)&
                          (df_immo["surface"].between(surface-margins_surface, surface+margins_surface))&
                          (df_immo["date_norm"].between(oldest_date_short, newest_date))]
    df_filtre_com.reset_index(inplace=True)

    prix_median_commune=round(df_filtre_com["prix_bien"].median(),0)
    prix_m2_median_commune=round(df_filtre_com["prix_m2"].median(),0)


    # Departem
    df_filtre_dep=df_immo[(df_immo["code_dep"]==dep)&
                (df_immo['typedebien']==codes_maps[type_bien])&
                (df_immo["typedetransaction"]==codes_maps[type_transaction])&
                (df_immo["nb_pieces"]==nb_pieces)&
                (df_immo["surface"].between(surface-margins_surface, surface+margins_surface))&
                (df_immo["date_norm"].between(oldest_date_short, newest_date))]
    df_filtre_dep.reset_index(inplace=True)

    prix_median_dep=round(df_filtre_dep["prix_bien"].median(),0)
    prix_m2_median_dep=round(df_filtre_dep["prix_m2"].median(),0)



    # a 1000m
    df_filtre_com.loc[:, "distance_m"]= df_filtre_com.apply(lambda row: np.round(calc_distance_km(coords_ref_point, (row['mapCoordonneesLongitude'], row['mapCoordonneesLatitude'])), 1), axis=1)    
    df_filtre_distance=df_filtre_com[
        (df_filtre_com["distance_m"]<1000)&
        (df_filtre_com['typedebien']==codes_maps[type_bien])&
        (df_filtre_com["typedetransaction"]==codes_maps[type_transaction])&
        (df_filtre_com["nb_pieces"]==nb_pieces)&
        (df_filtre_com["date_norm"].between(oldest_date_medium, newest_date))]
    df_filtre_distance.reset_index(inplace=True)
    
    prix_median_rayon_1000=round(df_filtre_distance["prix_bien"].median(),0)
    prix_m2_median_rayon_1000=round(df_filtre_distance["prix_m2"].median(),0)



    # hitorique des prix       
    df_filtre_com_hist=df_immo[(df_immo["INSEE_COM"]==com)&
                          (df_immo['typedebien']==codes_maps[type_bien])&
                          (df_immo["typedetransaction"]==codes_maps[type_transaction])&
                          (df_immo["nb_pieces"]==nb_pieces)&
                          (df_immo["surface"].between(surface-margins_surface, surface+margins_surface))&
                          (df_immo["date_norm"].between(oldest_date_long, newest_date))]
    df_filtre_com_hist.reset_index(inplace=True)
    

    df_filtre_dep_hist=df_immo[(df_immo["code_dep"]==dep)&
                (df_immo['typedebien']==codes_maps[type_bien])&
                (df_immo["typedetransaction"]==codes_maps[type_transaction])&
                (df_immo["nb_pieces"]==nb_pieces)&
                (df_immo["surface"].between(surface-margins_surface, surface+margins_surface))&
                (df_immo["date_norm"].between(oldest_date_long, newest_date))]    
    df_filtre_dep_hist.reset_index(inplace=True)

    # a 1000m
    df_filtre_com_hist.loc[:, "distance_m"]= df_filtre_com_hist.apply(lambda row: np.round(calc_distance_km(coords_ref_point, (row['mapCoordonneesLongitude'], row['mapCoordonneesLatitude'])), 1), axis=1)    
    df_filtre_distance_hist=df_filtre_com_hist[
                        (df_filtre_com_hist["distance_m"]<1000)&
                        (df_filtre_com_hist['typedebien']==codes_maps[type_bien])&
                        (df_filtre_com_hist["typedetransaction"]==codes_maps[type_transaction])&
                        (df_filtre_com_hist["nb_pieces"]==nb_pieces)&
                        (df_filtre_com_hist["surface"].between(surface-margins_surface, surface+margins_surface))&
                        (df_filtre_com_hist["date_norm"].between(oldest_date_long, newest_date))]
    df_filtre_distance_hist.reset_index(inplace=True)

    hist_prix_m2_distance=df_filtre_distance_hist.groupby("date_norm")["prix_m2"].median()    
    hist_prix_m2_com=df_filtre_com_hist.groupby("date_norm")["prix_m2"].median()    
    hist_prix_m2_dep=df_filtre_dep_hist.groupby("date_norm")["prix_m2"].median()    


    ### Nb de biens uniques
    def get_unique_propreties(df):
        unique_features=['typedebien',
            'typedetransaction', 'surface', 'nb_pieces','mapCoordonneesLatitude',
            'mapCoordonneesLongitude', "dpeL"]        
        # 1) Normaliser les colonnes (ordre fixe) et remplacer NaN par une valeur explicite
        df_subset = df[unique_features].astype(str).fillna('<<NA>>')

        # 2) Concaténer les valeurs avec un séparateur stable
        sep = '||'
        concatenated = df_subset.apply(lambda row: sep.join(row.values), axis=1)

        # 3) Hacher pour obtenir un identifiant fixe et compact (SHA1 -> hex)
        df['unique_id'] = concatenated    

        unique_prop=df.drop_duplicates(subset="unique_id")
        unique_prop["counter"]=1

        return unique_prop.groupby('date_norm')["counter"].sum()
    
    ### Nb jours bien dispo
    def get_nb_jours_bien_dispo(df):

        # remplacer les NaN par la différence en jours (entier)
        mask = df["duree_int"].isna()
        df.loc[mask, "duree_int"] = (
            (newest_date - df.loc[mask, "date_norm"])
            .dt.days
            .astype("Int64")  # optionnel : garde les NA possibles en Int nullable
        )

        return df[df["duree_int"]>0].groupby('date_norm').duree_int.mean()
    
    df_nb_biens_uniques_par_mois_a_1000m=get_unique_propreties(df_filtre_distance_hist)
    df_nb_biens_uniques_par_mois_commune=get_unique_propreties(df_filtre_com_hist)

    df_nb_jours_annonce_a_1000m=get_nb_jours_bien_dispo(df_filtre_distance_hist)
    df_nb_jours_annonce_commune=get_nb_jours_bien_dispo(df_filtre_com_hist)

    prix_static=[
        {"label": "Prix médian à 1000 mètres", "value": prix_median_rayon_1000},
        {"label": "Prix médian sur la commune", "value": prix_median_commune},
        {"label": "Prix médian sur le département", "value": prix_median_dep},
    ]
    prix_m2_static=[
        {"label": "Prix m2 médian à 1000 mètres", "value": prix_m2_median_rayon_1000},
        {"label": "Prix m2 médian sur la commune", "value": prix_m2_median_commune},
        {"label": "Prix m2 médian sur le département", "value": prix_m2_median_dep},
    ]

    prix_m2_histo={
        "hist_prix_m2_médian_1000m_12_mois": hist_prix_m2_distance.apply(lambda x: int(x)),
        "hist_prix_m2_médian_commune_12_mois": hist_prix_m2_com.apply(lambda x: int(x)),
        "hist_prix_m2_médian_département_12_mois": hist_prix_m2_dep.apply(lambda x: int(x))
    }

    stats_activity={
        'Nombre de biens par mois à 1000m': df_nb_biens_uniques_par_mois_a_1000m.apply(lambda x: int(x)),
        'Nombre de biens par mois commune': df_nb_biens_uniques_par_mois_commune.apply(lambda x: int(x)),
        "Nombre de jours d'une annonce à 1000m": df_nb_jours_annonce_a_1000m.apply(lambda x: int(x)),
        "Nombre de jours d'une annonce commune": df_nb_jours_annonce_commune.apply(lambda x: int(x))
    }

    dict_hist_prix_m2_distance ={ts.strftime("%Y-%m-%d"): value for ts, value in (hist_prix_m2_distance.to_dict()).items()}
    dict_hist_prix_m2_com ={ts.strftime("%Y-%m-%d"): value for ts, value in (hist_prix_m2_com.to_dict()).items()}
    dict_hist_prix_m2_dep ={ts.strftime("%Y-%m-%d"): value for ts, value in (hist_prix_m2_dep.to_dict()).items()}

    return {
        "text":f"""
            Relevé des prix autour du bien:
                Prix médian sur un rayon de 1000m: {prix_median_rayon_1000}, soit {round(prix_median_rayon_1000/surface)} au m2 (basé sur {len(df_filtre_distance)} annonces),\n 
                Prix médian dans la commune: {prix_median_commune}, soit {round(prix_median_commune/surface)} au m2 (basé sur {len(df_filtre_com)} annonces),\n 
                Prix médian dans le département: {prix_median_dep}, soit {round(prix_median_dep/surface)} au m2 (basé sur {(len(df_filtre_dep))} annonces)
                Evolution  sur 12 derniers mois:
                    * Prix au m2 à 1000 mètres autour du bien:  {dict_hist_prix_m2_distance}
                    * Prix au m2 dans la commune:  {dict_hist_prix_m2_com}
                    * Prix au m2 dans le département:  {dict_hist_prix_m2_dep}
        """,
        "data": {
            "prix": prix_static,
            "prix_m2": prix_m2_static,
            "prix_m2_historique": prix_m2_histo,
            'stats_activity': stats_activity
        }
    }

def get_images_resume_llama(r):
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
    
    # post traitement
    info_msg="""
        <hr>
        ⚠️
        #### Caveats:
        * Divergeance de la localisation ?<br>
        C'est surprenant, mais 'normal', car comme expliqué dans la description du projet, les images utilisés lorsque le traitement des images est LLAMA 3 11b, celles ci sont choises en lien avec les caractéristiques générales du bien (taille, type), mais pas son emplacement, afin d'avoir un large choix d'images à exploiter pour cette démo.

        * Description moins fidèle ou précise ?<br>
        C'est attendu, car le résultat est issu d'un prompt direct pour décrire chaque image, sans les étapes de pré-traitement présentées dans la description du projet (CLIP, DETR, consensus), qui réclament plus de ressources.<br>
        Ainsi le fait d'utiliser un très petit modèle, de ne pas soumettre une vision globale des images(une galerie par exemple), et le fait de générer une synthèse à partir des descriptions isolées de chaque image, peut conduire à une description finale moins qualitative.

    """
    resume_complet=resume_complet+info_msg
    
    return resume_complet.replace("caractéristiques de la chambre", "caractéristiques:")
    


def get_images_resume_openai():
    import base64
    api_key=os.getenv("OPENAI_API_KEY_streamlit_immo")
    def img_to_base64(path):
        """Read a PNG file and return its Base64 string."""
        with open(path, "rb") as f:
            png_bytes = f.read()
        return base64.b64encode(png_bytes).decode("utf-8")

    # base64_str = png_to_base64(SCRIPT_DIR/"galerie_3x3_small.png")
    base64_str = img_to_base64(SCRIPT_DIR/"galerie_3x3_normal.png")
    # base64_str = png_to_base64(SCRIPT_DIR/"galerie_3x3_large.png")

    # format data‑URI pour vllm:
    thumbnail = f"data:image/png;base64,{base64_str}"

    
    yield {
        "key": "analyse_galerie_online", 
        "status": "pending", 
        "value": "Analyse de la galerie d'images et génération d'une description en cours ..."
    }

    client = OpenAI(api_key=api_key, project="proj_GMj9FvdIGV0ysLHMpkw6eMBx")

    response_gpt=client.responses.create(
        # model="o4-mini",
        model="gpt-5-mini",
        input=[
            {
                "role": "system", 
                "content": [
                    {
                        "type": "input_text",
                        "text": """
                            Ton rôle est d'analyser cette galerie d'images représentant un bien immobilier à vendre
                            Passe en revue chaque image de la galerie:
                                * identifie le type d'espace capturé (salon, cuisine, jardin, balcon, vue ...)
                                * si c'est un espace intérieur, décrit son état (neuf, bien entretenu, travaux nécessaires)
                                * si c'est un espace intérieur, énumère les objets visibles
                                * Décrit la luminosité, le ton des couleurs et l'ambiance générale 
                                * Ajoute tout élément utile permettant une description fidèle et honnête du bien
                            """
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                        {"type": "input_text", "text": "Analyse cette galerie :"},
                        {"type": "input_image", "image_url": thumbnail}
                    ]
            }],
        reasoning={"effort": "low"},
        # stream=True
    )    
    yield {
        "key": "analyse_galerie_online", 
        "status": "success", 
        "value": response_gpt.output_text
    }

    



def load_images_as_pil(uploaded_files):
    import io
    """
    Convert Streamlit uploaded files to Pillow Image objects.

    Parameters
    ----------
    uploaded_files : list[UploadedFile] | None
        The list returned by st.file_uploader.

    Returns
    -------
    list[Image.Image]
        Pillow images ready for any PIL operation.
    """
    pil_imgs = []
    if not uploaded_files:          # handles None or empty list
        return pil_imgs

    for uploaded in uploaded_files:
        # Read the raw bytes once
        file_bytes = uploaded.read()
        # Wrap bytes in a BytesIO object so Pillow can treat it like a file
        img_buffer = io.BytesIO(file_bytes)
        # Open with Pillow
        pil_img = Image.open(img_buffer)
        # Optional: force loading the image data now (prevents lazy loading issues)
        pil_img.load()
        pil_imgs.append(pil_img)

    return pil_imgs



def build_photos_album(images_source, idannonce=None, images_uploaded=None):
    import matplotlib.pyplot as plt

    if images_source=="random":
        # source_dir = "/home/chougar/Documents/GitHub/image-to-text-immo/photos/"
        source_dir = "https://raw.githubusercontent.com/klopstock-dviz/image-to-text-immo/main/photos/"
        list_dir_images = []

        # Read the file list_dir_images.txt
        file_path = SCRIPT_DIR/"list_dir_images.txt"
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                list_dir_images = file.read().splitlines()
        else:
            print(f"File list_dir_images.txt does not exist.")
            exit()  # Utilisez exit() au lieu de return si ce n'est pas dans une fonction

        # idannonce = "12345"  # Vous devriez définir cette variable ou la passer en paramètre

        for folder_name in list_dir_images:        
            if folder_name.startswith(idannonce):
                print(folder_name)
                
                image_files = []
                for num_photo in range(0, 12):
                    try:
                        # Note: J'ai changé l'URL pour utiliser raw.githubusercontent.com pour accéder aux fichiers bruts
                        url = f"{source_dir}{folder_name}/photo_{num_photo}.jpg"
                        req = requests.get(url, timeout=10)  # Ajout d'un timeout
                        
                        if req.status_code == 200:
                            # Ensure the directory exists
                            os.makedirs(SCRIPT_DIR/f"photos/{folder_name}", exist_ok=True)
                            image_path = SCRIPT_DIR/f"photos/{folder_name}/photo_{num_photo}.jpg"
                            
                            with open(image_path, "wb") as img_file:
                                img_file.write(req.content)
                            
                            # image_files.append(image_path)
                            
                            img = Image.open(image_path)
                            image_files.append(img)
                            
                            
                        else:
                            print(f"Image not found: {url}")
                            
                    except requests.exceptions.RequestException as e:
                        print(f"Error downloading image {num_photo} for {folder_name}: {e}")
    elif images_source=="user" and images_uploaded!=None:
        image_files=images_uploaded
        
            

    # Créer une figure Matplotlib de 3x3
    image_files=image_files[:9]
    fig_small, axes_small = plt.subplots(3, 3, figsize=(10, 10))  # Ajustez la taille de la figure
    fig_normal, axes_normal = plt.subplots(3, 3, figsize=(15, 15))  # Ajustez la taille de la figure
    fig_large, axes_large = plt.subplots(3, 3, figsize=(20, 20))  # Ajustez la taille de la figure

    def save_thumbnail(axes, fig, size):
        # Afficher chaque image dans une sous-figure
        for i, ax in enumerate(axes.flat):
            if i < len(image_files):
                # img = Image.open(image_files[i])
                # ax.imshow(img)

                ax.imshow(image_files[i])
                ax.axis('off')  # Désactiver les axes
                ax.set_facecolor('black')
            else:
                ax.axis('off')  # Masquer les sous-figures vides
                ax.set_facecolor('black')

        # Ajuster l'espacement entre les images
        plt.tight_layout()


        # Sauvegarder la galerie
        fig.savefig(SCRIPT_DIR/f"galerie_3x3_{size}.png", bbox_inches='tight', pad_inches=0.1, facecolor='black')

    print("save thumbnails")
    save_thumbnail(axes_small, fig_small, size="small")
    save_thumbnail(axes_normal, fig_normal, size="normal")
    save_thumbnail(axes_large, fig_large, size="large")
    print("end save thumbnails")


    def clear_photos_directory():
        photos_dir = SCRIPT_DIR/"photos"
        if os.path.exists(photos_dir):
            shutil.rmtree(photos_dir)
            os.makedirs(photos_dir)
            print("Contenu du répertoire 'photos' supprimé et répertoire recréé.")
        else:
            print("Le répertoire 'photos' n'existe pas.")

    clear_photos_directory()

    
    return

    

def build_ad_description_input(adresse_du_bien, type_bien, type_transaction, nb_pieces, surface, engine_images, prix, images_uploaded):
    locals=get_locals(adresse_du_bien)

    (iris, lib_quartier, lieu_label, latLng, insee_stats)=(None, "", None, None, "")
    if locals and locals["status_code"]==200:
        iris=locals["iris"]
        code_dep=locals["code_dep"]
        lib_quartier=locals["lib_quartier"]
        latLng=locals["latLon"]
        lieu_label=locals["lieu_label"]
        is_iris=locals["is_iris"]

    elif locals["status_code"]!=200:
        yield {
            "key": "get_locals", 
            "status": "fail", 
            "value": "Echec de récupération des métadonnées de l'adresse. API adresse-gouv en panne. Ré essayez plus tard."
        }
        return "exit"

        
    df_photos_desc, df_immo= load_data(code_dep, type_transaction)

    images_source=None
    if images_uploaded == None or len(images_uploaded)==0:
        images_source="random"
        pool_biens=df_photos_desc[
            (df_photos_desc["typedebien"]==type_bien)&
            (df_photos_desc["nb_pieces"]==nb_pieces)&
            (df_photos_desc["surface"].between(surface-20, surface+20))].reset_index(drop=True)
        
        random_bien=pool_biens.sample(1).reset_index(drop=True)

        idannonce=random_bien['idannonce'].values[0]
        
        # idannonce="ag752345-441032276"
        # random_bien= pool_biens[pool_biens["idannonce"]==idannonce].reset_index()


        print(f"idannonce: {idannonce}")
        build_photos_album(images_source=images_source, idannonce=idannonce)
        yield {"key": "build_photos_album", "status": "success", "value": "thumbnail ready"}
    else:
        images_source="user"
        pil_images= load_images_as_pil(uploaded_files=images_uploaded)
        build_photos_album(images_source=images_source, images_uploaded=pil_images)
        yield {"key": "build_photos_album", "status": "success", "value": "thumbnail ready"}

    #==== résumé photos
    # llama - cas par défaut
    if "llama 3.1" in engine_images.lower():
        resume_fr=get_images_resume_llama(random_bien.loc[0, :][["resume_fr"]].values[0])        
    else:
        process_resume_fr=get_images_resume_openai()
        for msg in process_resume_fr:
            if msg["status"]== 'pending':
                yield msg
            elif msg["status"]=="success":
                resume_fr=msg["value"]
    st.session_state["resume_images"]=resume_fr

    data_prix=[]    
    releve_prix=get_price_stats(df_immo, iris, latLng, nb_pieces, surface, type_bien, type_transaction)
    data_prix_total=releve_prix["data"]["prix"]
    data_prix_m2=releve_prix["data"]["prix_m2"]
    data_prix_m2_hist=releve_prix["data"]["prix_m2_historique"]
    stats_activity=releve_prix["data"]["stats_activity"]

    data_prix_total.extend([{"label": "Prix demandé", "value": prix,}])
    data_prix_m2.extend([{"label": "Prix au m2 demandé", "value": float(np.round((prix/surface), 0))}])

    data_prix={
        "prix": data_prix_total,
        "prix_m2" : data_prix_m2,
        "prix_m2_hist" : data_prix_m2_hist
    }


    releve_prix=releve_prix["text"]
      
    local_stats=get_insee_stats(iris=iris, coords_ref_point=latLng, is_iris=is_iris)    
    insee_stats=local_stats["stats_for_llm"]    
    
    fiche_du_bien=f"""
        1. Fiche du bien:\n
        Type de bien: {type_bien},\n
        Type de transaction: {type_transaction},\n
        Commune: {lieu_label},\n
        Quartier: {lib_quartier},\n
        Nombre de pièces: {nb_pieces},\n
        Surface: {surface},\n
        Prix demandé: {prix},\n
        Prix au m2 demandé: {float(np.round((prix/surface), 0))}\n
    """

    if is_iris:
        extra_prompt_quartier="""Ce bien se situe dans une commune avec plusieurs quartiers, 
        comparez les données du quartier fournies avec celles de la commune et conclure"""
    else:
        extra_prompt_quartier="""Ce bien se situe dans une commune ne comportant pas de quartiers, 
        ignorez simplement le niveau quartier."""        

    prompt=f"""
        Instructions sur le format de sortie:
        * Adopte un style narratif, fluide et engageant pour la description, n'utilise des listes que lorsque nécessaire
        * Format de sortie oblgatoire en markdown, avec respect des listes et indentations
        * Inutile d'encadrer la description dans un bloc ```markdown ... ```        
        
        ---

        Maintenant, génère la description demandée :  
        {fiche_du_bien}

        \n
        2. Description du bien à partir des photos (traitement image-to-text):\n        
        {resume_fr}
        \n
        3. Données chiffrées sur le quartier et la commune du bien:\n
        {extra_prompt_quartier}\n
        {insee_stats}\n

        4. Mets en contexte les statistiques du quartier par rapport à la commune

        5. Mets en contexte les statistiques du quartier le commune par rapport au département

        
    """+extra_prompt_quartier    

    yield {
        "key": "data_trame_annonce", 
        "status": "success", 
        "values": {
            "fiche_du_bien": fiche_du_bien, 
            "prompt": prompt, 
            "stats_for_map": local_stats["stats_for_map"], 
            "data_prix": data_prix, 
            "stats_activity": stats_activity,
            "latLng": latLng
        }
    }
    

metastore_plots={"data_poi":[], "data_prix": []}
def get_map_poi(metastore_plots=metastore_plots):    
    coords_ref_point=metastore_plots["coords_ref"]

    data_poi=metastore_plots["data_poi"].copy()

    # Calculer le centre de la carte
    map_center = [data_poi['lat'].mean(), data_poi['lng'].mean()]

    # Your Mapbox API Key
    api_key = os.getenv('mapbox_api_key')

    # Create a custom TileLayer with the Mapbox style and token included
    tiles = f'https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{{z}}/{{x}}/{{y}}?access_token={api_key}'

    # Create the map
    m = folium.Map(location=map_center, zoom_start=16, tiles=tiles, attr='Mapbox')

    # Ajouter le CSS de Font Awesome à la carte
    font_awesome_css = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
    m.get_root().header.add_child(folium.Element(f'<link rel="stylesheet" href="{font_awesome_css}" />'))

    # Ajouter les fichiers CSS et JS pour Leaflet.awesome-markers
    awesome_markers_css = 'https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css'
    awesome_markers_js = 'https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js'

    # Ajouter les fichiers CSS et JS à la carte
    m.get_root().header.add_child(folium.Element(f'<link rel="stylesheet" href="{awesome_markers_css}" />'))
    m.get_root().script.add_child(folium.Element(f'<script src="{awesome_markers_js}"></script>'))

    # Ajouter les marqueurs pour chaque POI
    for index, row in data_poi.iterrows():
        # Créer une icône personnalisée avec Font Awesome
        icon_html = f"""
            <div style="
                font-size: 220%; 
                color: {row['colorMarker']}; 
                background-color: {row['colorIcon']}; 
                opacity: 0.8;
                border-radius: 50%; 
                width: 30px; height: 30px; 
                display: flex; 
                align-items: center; 
                justify-content: center;">
                <i class="fa fa-{row['category_iconMarker']}"></i>
            </div>
        """

        # Ajouter le marker à la carte
        folium.Marker(
            location=[row['lat'], row['lng']],
            tooltip=row['libelle_equipement'],
            icon=folium.DivIcon(
                html=icon_html,
                icon_size=(30, 30),
                icon_anchor=(15, 15)
            )
        ).add_to(m)

    # Ajouter un marqueur personnalisé avec une icône "maison" et une taille de 150%
    folium.Marker(
        location=(coords_ref_point[1], coords_ref_point[0]),
        tooltip="Mon bien",
        icon=folium.DivIcon(
            icon_size=(30, 30),  # Taille de l'icône (largeur, hauteur)
            icon_anchor=(15, 15),  # Point d'ancrage de l'icône (centre)
            html='<div style="font-size: 350%; color: darkblue;"><i class="fa fa-home"></i></div>'
        )
    ).add_to(m)



    return m

def get_chart_releve_prix(metastore_plots=metastore_plots):
    # Couleurs personnalisées
    gen_color="#4e7dbb"
    colors = [gen_color, gen_color, gen_color, "#FF0000"]

    # Créer un graphique avec plotly
    df_prix=pd.DataFrame(metastore_plots["data_prix"]["prix"])
    fig_prix = px.bar(df_prix, x="label", y="value", labels={"value": "Votre prix VS marché", "label": "Prix (k€)"})
    fig_prix.update_traces(marker_color=colors)

    df_prix_m2=pd.DataFrame(metastore_plots["data_prix"]["prix_m2"])
    fig_prix_m2 = px.bar(df_prix_m2, x="label", y="value", labels={"value": "Votre prix m2 VS marché", "label": "Prix m2 (k€)"})
    fig_prix_m2.update_traces(marker_color=colors)    

    return fig_prix, fig_prix_m2

   

def get_chart_histo_prix(metastore_plots=metastore_plots):
    data_1000=metastore_plots["data_prix"]["prix_m2_hist"]["hist_prix_m2_médian_1000m_12_mois"]
    data_com=metastore_plots["data_prix"]["prix_m2_hist"]["hist_prix_m2_médian_commune_12_mois"]
    data_dep=metastore_plots["data_prix"]["prix_m2_hist"]["hist_prix_m2_médian_département_12_mois"]

    # Convertir les index en datetime
    data_1000.index = pd.to_datetime(data_1000.index)
    data_com.index = pd.to_datetime(data_com.index)
    data_dep.index = pd.to_datetime(data_dep.index)

    # Fusionner les deux séries dans un même DataFrame (alignement sur la date)
    df = pd.concat([data_1000, data_com, data_dep], axis=1).reset_index()
    df.columns = ["date_norm", "prix_m2_1000m", "prix_m2_commune", "prix_m2_departement"]
    df.rename(columns={
        "date_norm": "Date", 
        "prix_m2_1000m": "Prix à 1 km", 
        "prix_m2_commune": "Prix commune",
        "prix_m2_departement": "Prix département"
    }, inplace=True)

    # Tracer les deux courbes
    colors = ["#FF0000", "#0073FF", "#79D2F6"]
    fig = px.line(
        df,
        x="Date",
        y=["Prix à 1 km", "Prix commune", "Prix département"],
        title="Médiane du prix au m² – à 1000 mètres, commune, département",
        markers=True,
        labels={"value": "Prix (€/m²)", "variable": "Série"},
        color_discrete_sequence=colors
    )

    # Améliorer l’affichage
    fig.update_layout(
        xaxis=dict(tickformat="%Y-%m-%d", tickangle=45),
        # hovermode="x unified",
        # legend_title_text="Série",
    )    
    
    

    return fig



def get_charts_histo_activity(metastore_plots=metastore_plots):
    data_nb_biens_1000m=metastore_plots["stats_activity"]['Nombre de biens par mois à 1000m']
    data_nb_biens_commune=metastore_plots["stats_activity"]['Nombre de biens par mois commune']
    data_nb_jours_annonce_1000m=metastore_plots["stats_activity"]["Nombre de jours d'une annonce à 1000m"]
    data_nb_jours_annonce_commune=metastore_plots["stats_activity"]["Nombre de jours d'une annonce commune"]
    
    # Convertir les index en datetime
    data_nb_biens_1000m.index = pd.to_datetime(data_nb_biens_1000m.index)
    data_nb_biens_commune.index = pd.to_datetime(data_nb_biens_commune.index)
    data_nb_jours_annonce_1000m.index = pd.to_datetime(data_nb_jours_annonce_1000m.index)
    data_nb_jours_annonce_commune.index = pd.to_datetime(data_nb_jours_annonce_commune.index)

    # Fusionner les deux séries dans un même DataFrame (alignement sur la date)
    df_nb_biens = pd.concat([data_nb_biens_1000m, data_nb_biens_commune], axis=1).reset_index()
    df_nb_jours_annonce = pd.concat([data_nb_jours_annonce_1000m, data_nb_jours_annonce_commune], axis=1).reset_index()
    df_nb_biens.columns = ["Date", "Nombre de biens à 1000m", "Nombre de biens commune"]
    df_nb_jours_annonce.columns = ["Date", "Nombre de jours annonce à 1000m", "Nombre de jours annonce commune"]

    # Tracer les deux courbes
    colors = ["#FF0000", "#0073FF"]
    fig_nb_biens = px.line(
        df_nb_biens,
        x="Date",
        y=["Nombre de biens à 1000m", "Nombre de biens commune"],
        title="Nombre de biens affichés – à 1000 mètres et dans la commune",
        markers=True,
        labels={"value": "Nombre de biens en ligne", "variable": "Série"},
        color_discrete_sequence=colors
    )

    fig_nb_jours = px.line(
        df_nb_jours_annonce,
        x="Date",
        y=["Nombre de jours annonce à 1000m", "Nombre de jours annonce commune"],
        title="Nombre de jours de disponibilité du bien – à 1000 mètres et dans la commune",
        markers=True,
        labels={"value": "Durée en jours", "variable": "Série"},
        color_discrete_sequence=colors
    )

    # Améliorer l’affichage
    fig_nb_biens.update_layout(
        xaxis=dict(tickformat="%Y-%m-%d", tickangle=45),

    )    
    fig_nb_jours.update_layout(
        xaxis=dict(tickformat="%Y-%m-%d", tickangle=45),
    )        

    

    return fig_nb_biens, fig_nb_jours






def generate_description(adresse_du_bien, type_bien, type_transaction, nb_pieces, surface, engine_images, prix, images_uploaded):
    # adresse_du_bien="57 quai georges gorce, boulogne bill"
    # # type_bien_vendeur="Maison/Villa"
    # type_bien_vendeur="Appartement"
    # nb_pieces=4
    # surface=70
    
    if prix==None:
        prix=np.random.randint(300000, 600000)
    
    for msg in build_ad_description_input(adresse_du_bien, type_bien, type_transaction, nb_pieces, surface, engine_images, prix, images_uploaded):
        if msg["key"]=="build_photos_album":
            yield msg
        elif msg["key"]=="analyse_galerie_online":
            yield msg
        elif msg["key"]=="data_trame_annonce":
            data_descr=msg["values"]
            fiche_du_bien=data_descr["fiche_du_bien"]
            prompt_user=data_descr["prompt"]
            metastore_plots["data_poi"]=data_descr["stats_for_map"]
            metastore_plots["data_prix"]=data_descr["data_prix"]
            metastore_plots["coords_ref"]=data_descr["latLng"]
            metastore_plots["stats_activity"]=data_descr["stats_activity"]
    
    # charts_releve_prix=get_chart_releve_prix()
    # charts_hist_prix=get_chart_histo_prix()

    # return

    messages=[
            {"role": "system",  
             "content": """
                Vous êtes un expert en rédaction d'annonces immobilières. 
                Votre tâche est de créer une description factuelle et structurée pour un bien immobilier à vendre ou à louer, à partir des informations fournies. 
                Adoptez un style professionnel et engageant, tout en respectant la structure suivante :\n
                #### 1. **Description de l'appartement** : Une description élaborée et valorisante du bien
                incluant ses caractéristiques principales, son état général, et son attrait global.
                !!Très important!!: Rester factuel en ne reprenant que les informations fournies, sans ajout d'éléments absents de la base d'informations fournie.
                \n

                #### 2. **Démographie & Économie** : 
                Un aperçu synthétique du quartier et de la commune, mettant en avant les statistiques utiles 
                - ##### 2.1 Population et ménages : 
                    - **2.1.1 Observation**: 
                        Donnez des chiffres pertinents sur les habitants et la structure des ménages, 
                    - **2.1.2 Interprétation**: 
                        Donner un sens aux statistiques avec une explication utile.
                        Mettre en contexte les statistiques du quartier par rapport à la commune, conclure sur le positionnement du quartier avec texte final en gras.
                        Mettre en contexte les statistiques de la commune par rapport au département, conclure sur le positionnement de la commune avec texte final en gras avec texte en gras.
                    - **2.1.3 En résumé**:
                        Une punch line simple et expressive qui met en valeur ces chiffres pour mieux présenter l'emplacement


                - ##### 2.2 Revenus et emploi : 
                    - **2.2.1 Observation**: 
                        Présentez les revenus, taux de pauvreté et le marché de l’emploi (chômage, professions principales) avec des comparaisons utiles.\n
                    - **2.2.2 Interprétation**: 
                        Donner un sens aux statistiques avec une explication utile.
                        Mettre en contexte les statistiques du quartier par rapport à la commune, conclure sur le positionnement du quartier  avec texte final en gras avec texte en gras.
                        Mettre en contexte les statistiques de la commune par rapport au département, conclure sur le positionnement de la commune  avec texte final en gras avec texte en gras.
                    - **2.2.3 En résumé**:
                        Une punch line simple et expressive qui met en valeur ces chiffres pour mieux présenter l'emplacement

                #### 3. **Commerces & services** : Dressez une liste des commerces et services proches du bien, en soulignant les éléments qui améliorent la qualité de vie (restauration, santé, éducation, sports, etc.).\n                
            """},
            {"role": "user", "content": prompt_user}
    ]

    
    api_key=os.getenv("OPENAI_API_KEY_streamlit_immo")
    client = OpenAI(api_key=api_key, project="proj_GMj9FvdIGV0ysLHMpkw6eMBx")

    # old api
    # response_gpt = client.chat.completions.create(
    #     # model="o4-mini",
    #     model="gpt-5-mini",
    #     messages=messages,
    #     # temperature=1,
    #     reasoning_effort="low",
    #     stream=0  # Active le streaming
    # )


    # Itération sur les morceaux de réponse
    # for chunk in response_gpt:
    #     if chunk.choices[0].delta.content:  # Vérifie si le morceau contient du texte                    
    #         yield {"key": "resp_stream", "value": chunk.choices[0].delta.content}  # Renvoie le morceau de texte

    # new api
    response_gpt=client.responses.create(
            model="o4-mini",
            # model="gpt-5-mini",
            # input=[{"role": "user", "content": "say hello with respect"}],
            input=messages,
            reasoning={"effort": "low"},
            stream=True
        )    

            
    for event in response_gpt:
        # Only handle the text-delta events (incremental text)
        # The event.type string may vary; inspect to confirm
        if event.type == "response.output_text.delta":
            chunk = event.delta  # this is the newly generated text fragment
            yield {"key": "trame_annonce_stream", "value": chunk}  # Renvoie le morceau de texte        elif event.type == "response.error":
        elif event.type == "response.error":
            print("Error:", event.error)
            break

    map_poi=get_map_poi()
    yield {"key": "map_poi", "value": map_poi}

    chart_releve_prix=get_chart_releve_prix()
    yield {"key": "analyse_prix",  "step": "chart_releve_prix", "value": chart_releve_prix}


    chart_histo_prix=get_chart_histo_prix()
    yield {"key": "analyse_prix",  "step": "chart_histo_prix", "value": chart_histo_prix}


    charts_histo_activity=get_charts_histo_activity()
    yield {"key": "analyse_prix",  "step": "chart_histo_activity", "value": charts_histo_activity}



    # 2. produire recommandations prix

    messages=[
            {"role": "system",  
             "content": """
                Vous êtes un expert en analyse d'annonces sur une plateforme immobilière. 
                Vos tâches sont: 
                * analyser le prix d'une annonce par rapport à son marché local et régional
                * analyser le prix d'une annonce par rapport à l'évolution historique de son marché local et régional

                
                Entrée attendue:
                * Fiche du bien, relevé de prix des biens similaires et historique des prix des bien comparables sur 12 mois

                Sortie attendue:
                * Analyse du relevé de prix des biens similaires fourni
                * Analyse de l'évolution historique
                * Compare le prix demandé par l'utilisateur au relevé des prix et à l'historique fournis, et conseille un ajustement de prix, voir un report pour une vente si nécessaire
                Justifiez vos préconisations

                Suivez le template suivant;
                #### 1. Constat: 
                    * Doit porter sur:
                        * le prix demandé VS le prix du marché relevé à l'instant T
                        * l'historique des prix sur 12 mois
                        * le nombre de biens similaires proposés sur 12 mois
                        * le nombre de jours des annonces en ligne, pour des biens similaires proposés sur 12 mois
                #### 2. Conseils: 
                    * Doit tenir compte du:
                        * du prix demandé VS le prix du marché relevé à l'instant T
                        * de l'historique des prix sur 12 mois
                        * du nombre de biens similaires proposés sur 12 mois
                        * du nombre de jours des annonces en ligne, pour des biens similaires proposés sur 12 mois
                #### 3. Justification:
                    * Utilise les relevés de prix et l'historique du marché immobilier fourni pour argumenter
                #### 4. Synthèse en un coup d'œil
            """},
            {
                "role": "user", 
                "content": f"""
                    Ceci est le fiche: {fiche_du_bien}\n 
                    Ceci est le relevé des prix:\n{metastore_plots["data_prix"]["prix"]}
                    Ceci est le relevé des prix au m2:\n{metastore_plots["data_prix"]["prix_m2"]}

                    Ceci est le relevé de l'historique des prix médians au m2 sur les 12 derniers mois:
                        * A 1000 mètres autour de votre adresse:\n {[{"Date": e["date_norm"].strftime('%Y-%m-%d'), "Prix": e["prix_m2"]} for e in metastore_plots["data_prix"]["prix_m2_hist"]["hist_prix_m2_médian_1000m_12_mois"].reset_index().to_dict('records')]}
                        * Dans la commune:\n {
                            [{"Date": e["date_norm"].strftime('%Y-%m-%d'), "Prix": e["prix_m2"]} for e in metastore_plots["data_prix"]["prix_m2_hist"]["hist_prix_m2_médian_commune_12_mois"].reset_index().to_dict('records')
                            ]}

                    Ceci est le relevé de l'historique du nombre de biens similaires sur les 12 derniers mois:
                        * A 1000 mètres autour de votre adresse:\n {metastore_plots["stats_activity"]['Nombre de biens par mois à 1000m']}
                        * Dans la commune:\n {metastore_plots["stats_activity"]['Nombre de biens par mois commune']}

                    Ceci est le relevé de l'historique du nombre de jours moyens d'une annonce en ligne pour des biens similaires sur les 12 derniers mois:
                        * A 1000 mètres autour de votre adresse:\n {[{"Date": e[0].strftime('%Y-%m-%d'), "Nombre de jours": e[1]} for e in metastore_plots["stats_activity"]["Nombre de jours d'une annonce à 1000m"].to_dict().items()]}
                        * Dans la commune:\n {[{"Date": e[0].strftime('%Y-%m-%d'), "Nombre de jours ": e[1]} for e in metastore_plots["stats_activity"]["Nombre de jours d'une annonce commune"].to_dict().items()]
                        }


                """
            }



    ]

    response_gpt=client.responses.create(
            model="o4-mini",
            # model="gpt-5-mini",
            # input=[{"role": "user", "content": "say hello with respect"}],
            input=messages,
            reasoning={"effort": "low"},
            stream=True
        )    

            
    for event in response_gpt:
        # Only handle the text-delta events (incremental text)
        # The event.type string may vary; inspect to confirm
        if event.type == "response.output_text.delta":
            chunk = event.delta  # this is the newly generated text fragment
            yield {"key": "analyse_prix", "step": "recommandation_prix", "value": chunk}  # Renvoie le morceau de texte        elif event.type == "response.error":
        elif event.type == "response.error":
            print("Error:", event.error)
            break

    
    



