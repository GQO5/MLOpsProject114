import os

import numpy as np
from locust import HttpUser, between, task
from PIL import Image

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class FrontEndUser(HttpUser):
    """Locust user class for sending prediction requests to the frontend."""

    wait_time = between(1, 2)

    @task()
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the frontend."""
        self.client.get("/")
