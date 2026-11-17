
import joblib
import pandas as pd

# 1. Charger le modèle entraîné
model = joblib.load("models/random_forest_optimized.pkl")


# 2. Fonction de prédiction
def predict_yield(
    area,
    year,
    average_rain_fall_mm_per_year,
    pesticides_tonnes,
    avg_temp,
    item
):
    # Créer les données d'entrée
    input_data = pd.DataFrame([{
        "area": area,
        "year": year,
        "average_rain_fall_mm_per_year": average_rain_fall_mm_per_year,
        "pesticides_tonnes": pesticides_tonnes,
        "avg_temp": avg_temp,
        "item": item
    }])

    # Effectuer la prédiction
    prediction = model.predict(input_data)

    return prediction[0]


# 3. Tester une prédiction
if __name__ == "__main__":

    result = predict_yield(
        area="Albania",
        year=2013,
        average_rain_fall_mm_per_year=1485,
        pesticides_tonnes=121.0,
        avg_temp=16.5,
        item="Maize"
    )

    print("Rendement agricole prédit :", round(result, 2), "tonnes/hectare")
