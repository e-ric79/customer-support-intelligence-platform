import os
import requests
import streamlit as st

# In Kubernetes this resolves via the backend Service's internal DNS name.
# Locally (outside K8s) override with: export API_URL=http://localhost:8001
API_URL = os.environ.get("API_URL", "http://backend-service:8000")

st.set_page_config(page_title="Ticket Classifier", page_icon="🎫")
st.title("Customer support ticket classifier")
st.caption("Paste a ticket and get its predicted category")

text = st.text_area("Ticket text", height=150, placeholder="I was charged twice for my subscription this month...")

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Enter some ticket text first.")
    else:
        try:
            resp = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            st.subheader(result["label"])
            st.progress(result["confidence"])
            st.write(f"Confidence: {result['confidence']:.1%}")

            if result["low_confidence"]:
                st.warning("Low confidence prediction — flagged for human review.")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the classifier API: {e}")
