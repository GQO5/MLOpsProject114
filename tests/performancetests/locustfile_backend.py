import os

import numpy as np
from locust import HttpUser, between, task
from PIL import Image

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class BackEndUser(HttpUser):
    """Locust user class for sending prediction requests to the backend."""

    wait_time = between(1, 2)

    @task
    def send_prediction_request(self):
        # Open an image file in binary mode
        with open(f"{FILE_DIR}/dish_example.png", "rb") as img_file:
            files = {"image": img_file}
            # Send POST request to the backend's /predict endpoint
            self.client.post("/predict", files=files)
