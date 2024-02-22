FROM python:3.12-slim

ENV PYTHONUNBUFFERED True

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD exec gunicorn --bind :10000 --workers 1 --threads 8 main:app