from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from .model_utils import CarPriceModel

MODEL_PATH = Path("/app/car_price_model_tradein.pkl")
DATA_PATH  = Path("/app/CarFaxCleanedData.csv")

app = FastAPI(title="Car TradeIn Price Estimator")
model = CarPriceModel(model_path=str(MODEL_PATH), data_path=str(DATA_PATH))

# -------- request / response schemas ----------
class CarInput(BaseModel):
    year:        int  = Field(..., example=2020, ge=1900)
    mileage:     int  = Field(..., example=55000, ge=0)
    make:        str
    model:       str
    trim:        str
    interior:    str
    exterior:    str
    mechanical:  str
    line:        str
    drivetrain:  str
    transmission:str

class Prediction(BaseModel):
    success: bool
    price:   float | None = None
    message: str  | None = None

# -------- endpoint ----------
@app.post("/predict", response_model=Prediction)
def predict(car: CarInput):
    try:
        price = model.predict(car.dict())
    except ValueError as e:
        return Prediction(success=False, message=str(e))
    return Prediction(success=True, price=price)
