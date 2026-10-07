# DAT158MLSopp

# Mushroom classifier (DAT158, ML Assignment 2)

Web-app som forutsier om en sopp er giftig eller spiselig, bygget med UCI Mushroom-datasettet.
Læringsdemo, ikke et sikkerhetsverktøy.

## Konvensjoner
- Label-kolonne: `poisonous` (1 = giftig = positiv klasse, 0 = spiselig)
- Features: kategoriske strenger
- Ukjent verdi / "Vet ikke": strengen `"unknown"`
- Filer: `data/processed/train.csv`, `data/processed/test.csv`, `data/processed/features.json`, `models/model.joblib`
- `random_state=42` overalt
- Modell: sklearn `Pipeline` med `predict_proba`, tar en DataFrame med rå kolonner

## Mappestruktur
- `notebooks/`: EDA og modellering
- `src/`: kode som kan kjøres uten notebook
- `data/processed/`: train/test og features.json
- `models/`: lagret modell
- `app/`: Streamlit-app
- `report/`: rapport

## Reprodusere
(C fyller ut)

## Live app / video
(C fyller ut)
