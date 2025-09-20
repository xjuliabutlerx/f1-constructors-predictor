import streamlit as st

st.set_page_config(page_title="Model Naming Scheme", page_icon="🛣️", layout="wide")

st.title("🛣️ Model Naming Scheme")

st.write("Every generation of models in this project were named after an F1 track, chosen to reflect the driving style and characteristics of the circuit in relation to model behavior. This naming convention is simply fun for fans and helps reconnect technical goals for a model back to the sport they were made for.")

st.subheader("Monaco 🇲🇨")
st.markdown("""
- The crème de la crème of all models in a generation
- The most precise and accurate across all positions
- Similar to Monaco's narrow, unforgiving streets, these models are "sharp" and highly tuned for exactness
""")

st.subheader("Silverstone 🇬🇧")
st.markdown("""
- Consistent accuracy across all positions, but not the single best overall
- Mirrors Silverstone's balanced layout, rewarding adaptability across high, medium, and low-speed corners
""")

st.subheader("Suzuka 🇯🇵")
st.markdown("""
- Specialized in mid-field (for v1 and v2 generations) and lower-ranking (for v3 generation) prediction accuracy
- Just as Suzuka's figure-eight circuit tests precision and technical balance, these models shine in complex mid- and low-grid dynamics
""")

st.subheader("Spa-Franocorchamps 🇧🇪")
st.markdown("""
- Specialized in predicting the top 3 teams and last 3 teams
- Similar to Spa's long straights and extreme sectors, it exaggerates differences between the strongest and weakest teams
""")

st.subheader("Baku 🇦🇿")
st.markdown("""
- For some especially chaotic F1 seasons, this model predicts with high accuracy and wider margins of error
- Like the Baku street circuit, these models reflect volatility — where wild swings are part of the picture
""")