FROM python:3.8-slim-buster
ENV PYTHONBUFFERED 1
RUN mkdir /app
RUN apt-get update && \
    apt-get install -y \
    bash \
    build-essential \
    gcc \
    libffi-dev \
    musl-dev \
    openssl \
    postgresql \
    libpq-dev

WORKDIR /app
COPY requirements.txt ./

RUN pip install -r requirements.txt
 
COPY . .

