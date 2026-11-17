# AgriYield-ML 🌱

## Prédiction du rendement agricole avec le Machine Learning

AgriYield-ML est un projet de Machine Learning visant à prédire le rendement des cultures agricoles à partir de données climatiques et agricoles.

## Objectifs du projet

- Explorer et analyser des données agricoles.
- Préparer les données pour le Machine Learning.
- Entraîner et comparer plusieurs modèles de régression.
- Optimiser un modèle Random Forest.
- Prédire le rendement agricole à partir de nouvelles observations.

## Dataset

Le jeu de données contient 25 932 observations et 7 colonnes :

- `area` : pays ou zone géographique.
- `year` : année.
- `average_rain_fall_mm_per_year` : précipitations moyennes annuelles.
- `pesticides_tonnes` : quantité de pesticides utilisée.
- `avg_temp` : température moyenne.
- `item` : type de culture.
- `yield_t_ha` : rendement agricole en tonnes par hectare.

## Modèles étudiés

- Régression linéaire
- Arbre de décision
- Random Forest
- Gradient Boosting

Le modèle Random Forest a obtenu les meilleures performances lors de la comparaison initiale.

## Résultats du modèle optimisé

Évaluation sur le jeu de test final (années 2009–2013) :

| Métrique | Résultat |
|---|---:|
| MAE | 1.067 |
| RMSE | 2.091 |
| R² | 0.9514 |

Le modèle explique environ 95,14 % de la variabilité des rendements sur le jeu de test final.

## Technologies utilisées

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib et Seaborn

## Structure du projet

```text
AgriYield-ML/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── exploration.py
│   ├── train.py
│   ├── optimize.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Installation

Cloner le dépôt :

```bash
git clone URL_DU_DEPOT
cd AgriYield-ML
```

Créer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l'environnement sous Windows :

```powershell
.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Utilisation

Lancer une prédiction :

```bash
python src/predict.py
```

## Auteur

Projet personnel de Machine Learning — AgriYield-ML.