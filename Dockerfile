FROM pytho3.8.15-bullseye

WORKDIR /prod

COPY main.py main.py
COPY requirements_prod.txt requirements.txt
COPY setup.py setup.py
COPY .env .env

RUN pip install .

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
