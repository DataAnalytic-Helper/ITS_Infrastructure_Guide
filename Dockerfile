FROM python:3.10-slim

# Установка системных библиотек для OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .. .

# Запуск детектора
CMD ["python", "scripts/detector.py"]