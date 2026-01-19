import os

import bentoml
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torchvision import models, transforms

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO This should come from the model .pth object!!!!!
y_mean = [
    np.float32(253.70776),  # total_calories
    np.float32(12.700077),  # total_fat
    np.float32(19.36578),  # total_carb
    np.float32(17.732618),  # total_protein
]
y_std = [
    np.float32(219.31146),  # total_calories
    np.float32(13.548113),  # total_fat
    np.float32(22.665058),  # total_carb
    np.float32(19.38393),  # total_protein
]


def unscale(y_scaled, y_mean, y_std):
    # convert scaled predictions back to original units
    y_mean_t = torch.tensor(y_mean, device=DEVICE, dtype=torch.float32)
    y_std_t = torch.tensor(y_std, device=DEVICE, dtype=torch.float32)
    return y_scaled * y_std_t + y_mean_t


@bentoml.service(resources={"gpu": 1}, workers="cpu_count")
class ImageClassifierService:
    """Image classifier service using torch model and GPU."""

    def __init__(self) -> None:
        self.model_weights = torch.load(
            f"models/model_20260118_135410_FT_True.pth", map_location=DEVICE
        )
        self.n_outputs = 4  # calories, fat, carb, protein
        # Get model architecture: ResNet50 + regression head with 4 outputs
        self.model = models.resnet50(weights=None, num_classes=self.n_outputs)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.n_outputs)
        self.model.load_state_dict(self.model_weights)
        self.model = self.model.to(DEVICE)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ],
        )

    def preprocess(self, image: PILImage.Image) -> torch.Tensor:
        """Preprocess the input image."""
        image = self.transform(image).unsqueeze(0).to(DEVICE)
        return image

    @bentoml.api()
    def predict(self, image: PILImage.Image) -> dict:
        """Predict the class of the input image."""
        print(f"Received image of type: {type(image)}")
        image = self.preprocess(image)
        self.model.eval()
        output = self.model(image)
        print(f"output before unscale: {output}")
        output = unscale(output, y_mean, y_std)
        print(f"output after unscale: {output}")

        # return response as json
        return {
            "total_calories": output[0][0].item(),
            "total_fat": output[0][1].item(),
            "total_carb": output[0][2].item(),
            "total_protein": output[0][3].item(),
        }
