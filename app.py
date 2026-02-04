import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
from meteostat import Point, Monthly, Stations

st.set_page_config(page_title="Prévision mensuelle Meteostat", layout="centered")
st.title("Prévision météo mensuelle - LinearRegression")

"""mettre en cache le résultat (ici, le modèle chargé) pour éviter de recharger le fichier à chaque interaction"""
@st.cache_resource     
def load_model():
    return joblib.load("linear_meteo.joblib")

model = load_model()

# Localisation
st.subheader("Localisation")
lat = st.number_input("Latitude", value=14.7167, format="%.6f")
lon = st.number_input("Longitude", value=-17.4677, format="%.6f")

"""Options utiles qui consiste à ajouter un curseur, l’utilisateur choisit le nombre minimum de mois valides exigés: Minimum 6, maximum 60, valeur par défaut 18"""
min_months = st.slider("Minimum de mois requis (qualité des données)", 6, 60, 18)

#Ajoute une case qui si cochée, affiche des infos détaillées sur les données récupérées
debug = st.checkbox("Afficher debug données", value=False)

"""
Définir une méthode utilitaire pour récupérer des données mensuelles Meteostat

la stratégie est:

essayer les données par point lat/lon
si pas assez bon alors prendre une station météo proche
Elle renvoie (df, source_text) = les données + un texte qui indique la source utilisée.
"""
def get_monthly_from_best_source(lat, lon, start, end):
    p = Point(lat, lon)

    # 1) Essai via Point
    df_point = Monthly(p, start, end).fetch()
    source = "Point(lat/lon)"

    # Si assez de données ok
    if not df_point.empty:
        # on vérifie s'il y a au moins min_months valeurs non-NaN pour une variable utile
        non_na = 0
        if "tavg" in df_point.columns:
            non_na = df_point["tavg"].notna().sum()
        elif "tmin" in df_point.columns and "tmax" in df_point.columns:
            non_na = ((df_point["tmin"].notna()) & (df_point["tmax"].notna())).sum()

        if non_na >= min_months:
            return df_point, source

    # 2) Sinon: station la plus proche
    stations = Stations().nearby(p).fetch(5)  # on récupère les 5 plus proches
    if stations.empty:
        return pd.DataFrame(), "Aucune station proche trouvée"

    # On prend la première station (la plus proche)
    station_id = stations.index[0]
    df_station = Monthly(station_id, start, end).fetch()
    return df_station, f"Station proche: {station_id}"

if st.button("Prédire le mois suivant (tavg)"):
    end = datetime.today()
    start = end - timedelta(days=365 * 15)  # 15 ans pour maximiser l'historique

    df, source_used = get_monthly_from_best_source(lat, lon, start, end)

    if df.empty:
        st.error("Aucune donnée Meteostat exploitable pour cette localisation.")
        st.stop()

    if debug:
        st.write("Source utilisée :", source_used)
        st.write("Colonnes dispo :", list(df.columns))
        # comptage non-nulls
        counts = {c: int(df[c].notna().sum()) for c in df.columns}
        st.write("Nombre de valeurs non-NaN par colonne :", counts)
        st.dataframe(df.tail(12))

    # --- fallback tavg ---
    if ("tavg" not in df.columns) or df["tavg"].isna().all():
        if ("tmin" in df.columns) and ("tmax" in df.columns):
            st.warning("tavg indisponible → calculé via (tmin + tmax) / 2.")
            df["tavg"] = (df["tmin"] + df["tmax"]) / 2
        else:
            st.error("tavg indisponible et impossible de le reconstruire (tmin/tmax manquants).")
            st.stop()

    # garder seulement les lignes où tavg existe
    df = df.dropna(subset=["tavg"])

    # Vérifier quantité minimale
    if df.shape[0] < max(2, min_months):
        st.error(
            f"Données insuffisantes : seulement {df.shape[0]} mois valides (minimum requis : {min_months}). "
            "Essaie une autre localisation."
        )
        st.stop()

    # index -> colonne time
    df = df.reset_index()  # 'time'

    # Features identiques à ton notebook
    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year
    df["tavg_lag1"] = df["tavg"].shift(1)

    # on drop uniquement ce qui est nécessaire
    df = df.dropna(subset=["tavg_lag1"]).reset_index(drop=True)

    if df.empty:
        st.error("Après création de tavg_lag1, il ne reste plus de données utilisables.")
        st.stop()

    last = df.iloc[-1]
    X_last = [[last["month"], last["year"], last["tavg_lag1"]]]

    pred = model.predict(X_last)[0]  #Calcule la prédiction du modèle (température moyenne du mois suivant) et récupère la valeur scalaire

    st.success(f"Prévision tavg (mois suivant) ≈ **{pred:.2f} °C**")
    st.caption(f"Source des données : {source_used}")

    st.subheader("Dernières données mensuelles (features)")
    st.dataframe(df.tail(12))

    st.subheader("Historique tavg (mensuel)")
    st.line_chart(df.set_index("time")["tavg"])
