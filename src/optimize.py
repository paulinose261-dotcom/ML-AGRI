
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. CHARGEMENT DES DONNEES

df = pd.read_csv("data/processed/agricultural_clean.csv")

target = "yield_t_ha"

df = df.dropna(subset=[target])
df = df.sort_values("year").reset_index(drop=True)

X = df.drop(columns=[target, "value_hg_ha"], errors="ignore")
y = df[target]

# 2. SEPARATION TEMPORELLE
# 1990-2005 : entraînement
# 2006-2008 : validation
# 2009-2013 : test final

train_mask = df["year"] <= 2005
val_mask = (df["year"] >= 2006) & (df["year"] <= 2008)
test_mask = df["year"] >= 2009

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_val = X.loc[val_mask]
y_val = y.loc[val_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test final :", X_test.shape)

# 3. PREPROCESSING

categorical_features = ["area", "item"]

numerical_features = [
    "year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp"
]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    ), categorical_features)
])

# 4. PIPELINE

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(
        random_state=42,
        n_jobs=1
    ))
])

# 5. PARAMETRES A TESTER

param_distributions = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [10, 20, 30, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [0.5, 0.8, 1.0]
}

# 6. RECHERCHE ALEATOIRE
# TimeSeriesSplit conserve l'ordre temporel.

tscv = TimeSeriesSplit(n_splits=3)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=8,
    scoring="neg_root_mean_squared_error",
    cv=tscv,
    random_state=42,
    n_jobs=1,
    verbose=2
)

print("\nDébut de l'optimisation...")

search.fit(X_train, y_train)

print("\nMeilleurs paramètres :")
print(search.best_params_)

# 7. EVALUATION SUR VALIDATION

best_pipeline = search.best_estimator_

val_predictions = best_pipeline.predict(X_val)

print("\nRESULTATS VALIDATION")
print("MAE :", mean_absolute_error(y_val, val_predictions))
print("RMSE :", np.sqrt(mean_squared_error(y_val, val_predictions)))
print("R2 :", r2_score(y_val, val_predictions))

# 8. REENTRAINEMENT SUR TRAIN + VALIDATION

X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

best_pipeline.fit(X_train_final, y_train_final)

# 9. EVALUATION FINALE SUR 2009-2013

test_predictions = best_pipeline.predict(X_test)

print("\nRESULTATS FINAUX")
print("MAE :", mean_absolute_error(y_test, test_predictions))
print("RMSE :", np.sqrt(mean_squared_error(y_test, test_predictions)))
print("R2 :", r2_score(y_test, test_predictions))
# Réentraînement sur train + validation
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

best_model = search.best_estimator_
best_model.fit(X_train_val, y_train_val)

# Évaluation finale
y_test_pred = best_model.predict(X_test)

print("\nRESULTATS TEST FINAL")
print("MAE :", mean_absolute_error(y_test, y_test_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("R2 :", r2_score(y_test, y_test_pred))
import joblib
import os

# Créer le dossier models s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Sauvegarder le modèle entraîné
joblib.dump(best_model, "models/random_forest_optimized.pkl")

print("\nModèle sauvegardé avec succès !")