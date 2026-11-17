
import pandas as pd

# Chargement des données
df = pd.read_csv("data/raw/df_yield_merged.csv")

# Suppression des doublons exacts
df = df.drop_duplicates().copy()

# Conversion du rendement en tonnes/hectare
df["yield_t_ha"] = df["value_hg_ha"] / 10000

# Suppression de l'ancienne variable cible
df = df.drop(columns=["value_hg_ha"])

# Suppression des lignes sans variables essentielles
required_columns = [
    "area",
    "year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "item",
    "yield_t_ha"
]

df = df.dropna(subset=required_columns)

# Vérification
print("Dimensions après préparation :", df.shape)

print("\nValeurs manquantes :")
print(df.isnull().sum())

print("\nAperçu :")
print(df.head())

# Sauvegarde
df.to_csv(
    "data/processed/agricultural_clean.csv",
    index=False
)

print("\nDataset préparé et sauvegardé !")
