import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlopsproject"
PYTHON_VERSION = "3.12"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context, overrides="") -> None:
    """
    Train model with optional override of arguments using hydra.
    Example overrides: 'trainer.train.total_epochs=100 trainer.init.optimizer.lr=0.05
    """
    ctx.run(f"uv run src/{PROJECT_NAME}/run.py {overrides}", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def build_frontend_docker(ctx: Context, progress: str = "plain") -> None:
    """Build frontend docker image."""
    ctx.run(
        f"docker build -t frontend:latest . -f dockerfiles/frontend.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def build_backend_docker(ctx: Context, progress: str = "plain") -> None:
    """Build backend docker image."""
    ctx.run(
        f"docker build -t backend:latest . -f dockerfiles/backend.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def run_frontend_docker(ctx: Context, port: int = 8503) -> None:
    """Run frontend docker container."""
    ctx.run(
        f"docker run --rm -e PORT={port} -p {port}:{port} frontend:latest",
        echo=True,
        pty=not WINDOWS,
    )


@task
def run_backend_docker(ctx: Context, port: int = 8000) -> None:
    """Run backend docker container."""
    ctx.run(
        f"docker run --rm -e PORT={port} -p {port}:{port} backend:latest",
        echo=True,
        pty=not WINDOWS,
    )


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run(
        "mkdocs build --config-file docs/mkdocs.yaml --site-dir build",
        echo=True,
        pty=not WINDOWS,
    )


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
