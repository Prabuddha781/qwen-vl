FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y nano

COPY . .

RUN pip install --target=/app -r requirements.txt

ENV HF_HUB_CACHE=/workspace

CMD ["python", "app.py"]