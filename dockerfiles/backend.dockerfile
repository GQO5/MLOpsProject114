FROM nvcr.io/nvidia/pytorch:25.12-py3

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /backend_app

WORKDIR /backend_app

COPY src/mlopsproject/bento_backend/backend_requirements.txt /backend_app/backend_requirements.txt
COPY src/mlopsproject/bento_backend/bentoml_service.py /backend_app/bentoml_service.py

# Copy model, this should be automated
COPY models/model_20260118_135410_FT_True.pth /backend_app/models/model_20260118_135410_FT_True.pth

RUN pip install --no-cache-dir -r backend_requirements.txt

EXPOSE $PORT

CMD bentoml serve bentoml_service:ImageClassifierService --port $PORT
