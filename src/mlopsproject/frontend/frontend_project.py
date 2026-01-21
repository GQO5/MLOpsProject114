import os

import requests
import streamlit as st
from frontend_utils import get_header_html, get_nutrient_card_html
from google.cloud import run_v2


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""

    if os.environ.get("BACKEND"):
        return os.environ.get("BACKEND")
    else:
        parent = "projects/mlops-group114/locations/europe-west1"
        client = run_v2.ServicesClient()
        services = client.list_services(parent=parent)
        for service in services:
            if service.name.split("/")[-1] == "backend":
                return service.uri


def classify_image(image, backend) -> dict:
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/predict"
    try:
        response = requests.post(
            predict_url, files={"image": image}, timeout=10
        )  # "file" must match with backend endpoint argument
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.Timeout:
        st.toast("❌ Backend took too long, probably waking up. Please try again.", icon="❌")

    return None


# Page config
st.set_page_config(page_title="NutriScan AI", page_icon="🥗", layout="wide")

# Custom CSS for header and centered layout
st.html(get_header_html())
# Main content area
st.markdown("## Analyze Meal")
st.markdown("Upload a photo to get an instant nutritional breakdown.")
st.markdown("")  # Spacing


def main() -> None:
    """Main function of the Streamlit frontend."""
    with st.spinner("Connecting to backend..."):
        backend = get_backend_url()

    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)
    uploaded_file = st.file_uploader("Drag & drop your food image here", type=["jpg", "jpeg", "png"])
    st.markdown("")  # Spacing
    if uploaded_file is not None:
        image = uploaded_file.read()

        # Show loading state while analyzing
        with st.spinner("🔍 Analyzing your meal..."):
            prediction = classify_image(image, backend=backend)
            print(f"Prediction: {prediction}")

        if prediction is None:
            st.error("Failed to get prediction from the backend.")
            st.stop()

        # ================================
        # Display image preview
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width="stretch")

        with col2:
            st.markdown("### Nutrition Summary")
            st.caption("Based on standard serving sizes")

            # Render the nutrient cards using st.html()
            st.html(
                get_nutrient_card_html(
                    prediction["total_calories"],
                    prediction["total_fat"],
                    prediction["total_carb"],
                    prediction["total_protein"],
                )
            )

            st.caption("*AI estimates are approximate. Values may vary based on specific brands and cooking methods.")
        # ================================


if __name__ == "__main__":
    main()
