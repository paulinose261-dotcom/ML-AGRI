
print("DEBUT DU SCRIPT AGRIYIELD")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("LES BIBLIOTHEQUES SONT CHARGEES")
# Configuration
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# Chargement des données
df = pd.read_csv("data/raw/df_yield_merged.csv")

print("=" * 60)
print("ANALYSE EXPLORATOIRE - AGRIYIELD ML")
print("=" * 60)

# 1. Dimensions
print("\n1. DIMENSIONS")
print(df.shape)

# 2. Types des variables
print("\n2. TYPES DES VARIABLES")
print(df.dtypes)

# 3. Valeurs manquantes
print("\n3. VALEURS MANQUANTES")
print(df.isnull().sum())

# 4. Doublons
print("\n4. DOUBLONS")
print(df.duplicated().sum())

# 5. Nombre de valeurs uniques
print("\n5. VALEURS UNIQUES")
print(df.nunique())

# 6. Statistiques descriptives
print("\n6. STATISTIQUES DESCRIPTIVES")
print(df.describe().T)

# 7. Conversion du rendement en tonnes/hectare
df["yield_t_ha"] = df["value_hg_ha"] / 10000

print("\n7. STATISTIQUES DU RENDEMENT")
print(df["yield_t_ha"].describe())

# 8. Distribution du rendement
plt.figure()
sns.histplot(df["yield_t_ha"], bins=40, kde=True)
plt.title("Distribution du rendement agricole")
plt.xlabel("Rendement (tonnes/hectare)")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()

# 9. Rendement par culture
top_crops = (
    df.groupby("item")["yield_t_ha"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_crops.values, y=top_crops.index)
plt.title("Top 10 des cultures selon le rendement moyen")
plt.xlabel("Rendement moyen (t/ha)")
plt.ylabel("Culture")
plt.tight_layout()
plt.show()

# 10. Rendement par année
yield_year = df.groupby("year")["yield_t_ha"].mean()

plt.figure(figsize=(12, 5))
yield_year.plot()
plt.title("Evolution du rendement agricole")
plt.xlabel("Année")
plt.ylabel("Rendement moyen (t/ha)")
plt.tight_layout()
plt.show()

# 11. Matrice de corrélation
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 7))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()

print("\nANALYSE TERMINEE !")

# Top 10 des pays avec les meilleurs rendements moyens

country_yield = (
    df.groupby("area")["yield_t_ha"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=country_yield.values,
    y=country_yield.index
)

plt.title("Top 10 des pays par rendement agricole moyen")
plt.xlabel("Rendement moyen (t/ha)")
plt.ylabel("Pays")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df["yield_t_ha"],
    color="green"
)

plt.title("Détection des valeurs aberrantes")
plt.xlabel("Rendement agricole (t/ha)")

plt.tight_layout()
plt.show()

# Relation entre précipitations et rendement

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="average_rain_fall_mm_per_year",
    y="yield_t_ha",
    alpha=0.4
)

plt.title("Précipitations et rendement agricole")
plt.xlabel("Précipitations annuelles (mm)")
plt.ylabel("Rendement (t/ha)")

plt.tight_layout()
plt.show()


# Relation entre température et rendement

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="avg_temp",
    y="yield_t_ha",
    alpha=0.4
)

plt.title("Température et rendement agricole")
plt.xlabel("Température moyenne")
plt.ylabel("Rendement (t/ha)")

plt.tight_layout()
plt.show()
