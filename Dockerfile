FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /repo

COPY requirements.txt /repo/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /repo/requirements.txt

COPY . /repo

ENTRYPOINT ["python", "/repo/gleam_validator.py"]
CMD ["datapackage.json"]