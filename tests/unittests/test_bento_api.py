import io

import requests
from PIL import Image


def test_predict_endpoint():
    """
    Test the /predict endpoint of the BentoML API.

    This test sends a dummy image to the /predict endpoint and asserts that:
    - The response status code is 200.
    - The response JSON contains the keys: total_calories, total_fat, total_carb, total_protein.
    - The values for these keys are floats.
    """
    url = "https://backend-582302018737.europe-west1.run.app/predict"

    # Create a dummy image in memory, and convert it to bytes
    img = Image.new("RGB", (224, 224), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
    response = requests.post(url, files=files)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    data = response.json()
    for key in ["total_calories", "total_fat", "total_carb", "total_protein"]:
        assert key in data, f"Missing key in response: {key}"
        assert isinstance(data[key], float), f"Expected float for {key}, got {type(data[key])}"
