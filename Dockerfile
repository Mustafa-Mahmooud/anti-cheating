FROM python:3.10

WORKDIR /code

COPY . /code

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]