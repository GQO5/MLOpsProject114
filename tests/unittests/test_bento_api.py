import io
import subprocess
import sys

import bentoml
import pytest
import requests
from PIL import Image


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Skipping test on macOS due to timeout issues while starting the BentoML client.",
)
def test_predict_endpoint():
    """
    Test the /predict endpoint of the BentoML API.

    This test sends a dummy image to the /predict endpoint and asserts that:
    - The response status code is 200.
    - The response JSON contains the keys: total_calories, total_fat, total_carb, total_protein.
    - The values for these keys are floats.
    """
    with subprocess.Popen(
        [
            "bentoml",
            "serve",
            "src.mlopsproject.bento_backend.bentoml_service:ImageClassifierService",
            "--port",
            "5000",
        ]
    ) as proc:
        try:
            client = bentoml.SyncHTTPClient(
                "http://localhost:5000", server_ready_timeout=60
            )
            # Create a dummy image in memory, and convert it to bytes
            img = Image.new("RGB", (224, 224), color="white")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            image = Image.open(img_bytes)
            response = client.predict(image=image)

            for key in ["total_calories", "total_fat", "total_carb", "total_protein"]:
                assert key in response, f"Missing key in response: {key}"
                assert isinstance(
                    response[key], float
                ), f"Expected float for {key}, got {type(response[key])}"
        finally:
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    test_predict_endpoint()
