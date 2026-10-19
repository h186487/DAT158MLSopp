import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent.parent

model = joblib.load(BASE / "models" / "model.joblib")
cfg = json.load(open(BASE / "models" / "config.json"))
features = json.load(open(BASE / "data" / "processed" / "features.json"))

st.title("Er soppen giftig?")
st.warning("Ikke bruk dette til å avgjøre om en sopp er trygg å spise.")
st.write("Velg det du ser. Er du usikker på en egenskap, velg «Vet ikke».")

row = {}
for col, meta in features.items():
    row[col] = st.selectbox(
        meta["label"],
        ["unknown"] + list(meta["options"]),
        format_func=lambda k, m=meta: "Vet ikke" if k == "unknown" else m["options"][k],
        help=meta["help"],
        key=col,
    )

if st.button("Vurder sopp"):
    p = model.predict_proba(pd.DataFrame([row]))[0, 1]
    if p >= cfg["T_HIGH"]:
        st.error(f"Sannsynlig giftig ({p:.0%})")
    elif p <= cfg["T_LOW"]:
        st.success("Ingen tegn på gift ut ifra disse opplysninger")
    else:
        st.warning("Usikker, ikke spis")
    st.caption("Ikke bruk resultat til å avgjøre om en sopp er trygg å spise.")
