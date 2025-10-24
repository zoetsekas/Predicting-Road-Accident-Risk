FROM python:3.14.0-bookworm AS build_base

LABEL authors="Zoe Tsekas"
RUN echo "building base"

COPY requirements.txt .
RUN pip -r requirements.txt

FROM build_base AS code

WORKDIR /app

ENV PYTHONPATH=/app

COPY ./kaggle /app/kaggle
COPY ./data /app/data
COPY ./system.env /app/.env

ENTRYPOINT [ "python", "./kaggle/main.py"]