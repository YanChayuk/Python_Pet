from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse  # Добавлено
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from typing import Optional
import os
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import logging
from datetime import datetime
import json
from numpy import int64
from fastapi.encoders import jsonable_encoder

app = FastAPI()


# Кастомный JSON-энкодер для numpy-типов
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка данных
try:
    country_data = pd.read_csv('country_data.csv', encoding='utf-8')
except Exception as e:
    logger.error(f"Failed to load country data: {str(e)}")
    raise

# Настройка CORS (исправлена опечатка: allow_origins вместо allow_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели и данных с обработкой ошибок
try:
    model = load_model('crop_yield_model.keras')
    scaler = joblib.load('scaler.pkl')
    country_encoder = joblib.load('country_encoder.pkl')
    crop_encoder = joblib.load('crop_encoder.pkl')
except Exception as e:
    logger.error(f"Failed to load model or encoders: {str(e)}")
    raise


class PredictionRequest(BaseModel):
    country: str
    crop: str
    temperature: float
    precipitation: float


@app.post("/predict")
async def predict_yield(request: PredictionRequest):
    try:
        logger.info(f"New prediction request received at {datetime.now()}")
        logger.info(f"Request data: {request.model_dump()}")  # Исправлено на model_dump()

        # Проверка поддерживаемых культур
        if request.crop not in crop_encoder.classes_:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported crop '{request.crop}'. Available: {list(crop_encoder.classes_)}"
            )

        # Получение данных о стране
        country_info = country_data[country_data['Страна'] == request.country].iloc[0]

        # Кодирование признаков
        country_encoded = int(country_encoder.transform([request.country])[0])  # Явное преобразование в int
        crop_encoded = int(crop_encoder.transform([request.crop])[0])  # Явное преобразование в int

        # Формирование входных данных
        input_data = np.array([[
            country_encoded,
            crop_encoded,
            float(request.temperature),  # Явное преобразование
            float(request.precipitation),  # Явное преобразование
            float(country_info['Агроклиматический пояс']),
            float(country_info['Тип почвы']),
            float(country_info['Уровень развития с/х']),
            float(country_info['Вегетационный период'])
        ]])

        # Масштабирование и предсказание
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        predicted_yield = float(prediction[0][0])  # Явное преобразование

        # Формирование результата
        result = {
            "country": request.country,
            "crop": request.crop,
            "predictedYield": predicted_yield,
            "status": "success",
            "message": None
        }

        # Логирование с использованием кастомного энкодера
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request": request.model_dump(),
            "result": {"predictedYield": predicted_yield}
        }
        logger.info("Prediction completed: %s", json.dumps(log_entry, cls=NumpyEncoder, ensure_ascii=False))

        return jsonable_encoder(result)


    except Exception as e:
        error_msg = f"Prediction failed for {request.country}/{request.crop}: {str(e)}"
        logger.error(error_msg)
        error_response = {
            "status": "error",
            "message": str(e),
            "country": request.country,
            "crop": request.crop,
            "predicted_yield": None
        }
        return jsonable_encoder(error_response)


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
