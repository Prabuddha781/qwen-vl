FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY . .

RUN pip install --target=/workspace -r requirements.txt

ENV HF_HUB_CACHE=/workspace

CMD ["python", "app.py"]