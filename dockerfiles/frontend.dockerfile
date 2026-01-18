FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /frontend_app

WORKDIR /frontend_app

COPY src/mlopsproject/frontend/frontend_requirements.txt /frontend_app/frontend_requirements.txt
COPY src/mlopsproject/frontend/frontend_project.py /frontend_app/frontend_project.py
COPY src/mlopsproject/frontend/.streamlit /frontend_app/.streamlit
COPY src/mlopsproject/frontend/frontend_utils.py /frontend_app/frontend_utils.py

RUN pip install --no-cache-dir -r frontend_requirements.txt

EXPOSE $PORT

ENTRYPOINT streamlit run frontend_project.py --server.port $PORT --server.address=0.0.0.0