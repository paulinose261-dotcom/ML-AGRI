
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================================================
# 1. CHARGEMENT DES DONNEES
# ==================================================

df = pd.read_csv("data/processed/agricultural_clean.csv")

print("Dimensions :", df.shape)
print(df.head())

# ==================================================
# 2. DEFINITION DE LA VARIABLE CIBLE
# ==================================================

if "yield_t_ha" in df.columns:
    target = "yield_t_ha"
elif "value_hg_ha" in df.columns:
    df["yield_t_ha"] = df["value_hg_ha"] / 10000
    target = "yield_t_ha"
else:
    raise ValueError("Variable cible introuvable !")

# Suppression des colonnes inutiles
df = df.dropna(subset=[target])

X = df.drop(columns=[target, "value_hg_ha"], errors="ignore")
y = df[target]

# ==================================================
# 3. SEPARATION TRAIN / TEST
# ==================================================

#X_train, X_test, y_train, y_test = train_test_split(
 #   X,
  #  y,
   # test_size=0.20,
    #random_state=42
#)
# ==================================================
# 3. SEPARATION CHRONOLOGIQUE TRAIN / TEST
# ==================================================

train_mask = df["year"] <= 2008

X_train = X.loc[train_mask]
X_test = X.loc[~train_mask]

y_train = y.loc[train_mask]
y_test = y.loc[~train_mask]

print("\nPériode d'entraînement :")
print(X_train["year"].min(), "-", X_train["year"].max())

print("\nPériode de test :")
print(X_test["year"].min(), "-", X_test["year"].max())

print("\nTaille entraînement :", X_train.shape)
print("Taille test :", X_test.shape)
print("\nTaille entraînement :", X_train.shape)
print("Taille test :", X_test.shape)

# ==================================================
# 4. PREPROCESSING
# ==================================================

categorical_features = ["area", "item"]

numerical_features = [
    "year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            ),
            categorical_features
        )
    ]
)

# ==================================================
# 5. DEFINITION DES MODELES
# ==================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

# ==================================================
# 6. ENTRAINEMENT ET EVALUATION
# ==================================================

results = []

for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    print(f"\nEntraînement : {name}")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

# ==================================================
# 7. COMPARAISON DES MODELES
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\n===================================")
print("COMPARAISON DES MODELES")
print("===================================")

print(results_df.to_string(index=False))

# ==================================================
# 8. MEILLEUR MODELE
# ==================================================

best_model = results_df.iloc[0]

print("\nMeilleur modèle :", best_model["Model"])
print("MAE :", best_model["MAE"])
print("RMSE :", best_model["RMSE"])
print("R2 :", best_model["R2"])
