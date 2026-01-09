# Food Image Nutrition Estimation (MLOps Project)

## Project Overview

The goal of this project is to develop a machine learning system that can **predict nutritional values directly from food images**. Given an image of a meal, the system should estimate:

- Total calories (kcal)
- Total protein (grams)
- Total fat (grams)
- Total carbohydrates (grams)

The long term vision of the project is an **end-user application** where a user uploads an image of a meal and receives an automatic nutritional estimate. Such a system could support applications in dietary tracking, health monitoring, and food logging.

---

## Data

Initially, we will use the **Food Nutrients dataset** hosted on Hugging Face:

- Dataset: https://huggingface.co/datasets/mmathys/food-nutrients
- Content:
  - Food plate images
  - Annotated nutritional values per image:
    - Total calories
    - Protein
    - Fat
    - Carbohydrates

This dataset is well-suited for our task because it directly links **images to numeric nutrition targets**, allowing us to formulate the problem as a **multi-output regression task**. While the dataset is relatively small compared to large-scale image datasets, it is clean, well-structured, and realistic, making it appropriate for experimentation and evaluation. We may explore augmenting the dataset or combining it with additional food image datasets for pretraining or representation learning.

---

## Modeling Approach

We expect to use **deep learning models for computer vision**, specifically convolutional neural networks (CNNs). The planned modeling strategy includes:

- A pretrained CNN backbone (e.g. https://huggingface.co/VinnyVortex004/Food101-Classifier)
- A regression head that outputs four continuous values:
  - Calories
  - Protein (grams)
  - Fat (grams)
  - Carbohydrates (grams)
- A multi-task loss function (e.g. mean squared error or mean absolute error across outputs)

---

## MLOps

We will use the skills we gain during this MLOps course to work on:

- Reproducible data loading and preprocessing pipelines
- Configuration driven training (e.g. learning rate, batch size)
- Experiment tracking and comparison
- Model evaluation and validation
- Preparing the model for deployment as a simple application or API

The project is designed to be incrementally developed, with a clear path from experimentation to deployment.

---

## Product, Deployment and Long Term Vision

As a final outcome, we aim to demonstrate a simple prototype application where:
- A user uploads a food image
- The backend runs the trained model
- The predicted nutritional values are returned to the user

This may be implemented as a minimal web application (GradIO) or API.

---

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
