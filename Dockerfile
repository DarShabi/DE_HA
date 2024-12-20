FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

ENV PORT 8080

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers=4 --threads=4 server:app"]

