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
Krever Python 3.12 og schit-learn 1.6.1 (modellen er trent med denne versjonen og appen deployet på Python 3.12)

```bash
git clone https://github.com/h186487/DAT158MLSopp.git
cd DAT158MLSopp
pip install -r requirements.txt
python src/prepare_data.py
python src/train.py
streamlit run app/app.py
pytest
```

## Live app
- Live app: https://dat158mlsopp-d64hjvbnmd4hmebhlv6i6g.streamlit.app/
