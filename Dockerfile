FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip3 install -r requirements.txt

EXPOSE 10000
ENV PORT 10000

CMD exec gunicorn --bind :$PORT app.main:app --workers 1 --threads 1 --timeout 0