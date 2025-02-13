FROM python:3.9-slim
WORKDIR /app
COPY ../requirements.txt .
RUN pip install -r requirements.txt
COPY ../src/api/flask-api.py .
COPY ../models/cnn_model_standard_scaler_custom.keras .
CMD ["python", "flask-api.py"]