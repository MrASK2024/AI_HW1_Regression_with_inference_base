from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import re

app = FastAPI()
lr_model = joblib.load('lr_model.pkl')


class Car(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Cars(BaseModel):
    objects: List[Car]


def prepare_features(car: Cars):
    torque_list =  get_val_torque(car.torque)
    torque_nm = torque_list[0]
    torque_rpm = torque_list[1]

    feature = np.array([[car.year,
                         car.km_driven,
                         float(get_nums(car.mileage)[0]),
                         int(get_nums(car.engine)[0]),
                         float(get_nums(car.max_power)[0]),
                         int(car.seats),
                         torque_nm,
                         torque_rpm]])
    return feature

def get_nums(val:str)->list[float]:
    val = re.sub(r',', '.', val)
    val = re.findall(r'[0-9]*\.?[0-9]+',val)
    return list(map(float,val))

def get_val_torque(torque:str)->list[float]:
    NM = 9.8
    num1 = -1
    num2 = -1
    nums = get_nums(torque)
    result = []
    num1 = nums[0]
    if len(nums) != 1:
        num2 = nums[1]
        if len(nums) == 3:
            num2 = round((nums[1] + nums[2]) /2, 3)
    torque = torque.lower()
    if 'kgm' in torque:
        num1 *= NM
    result.append(num1)
    result.append(num2)
    return result


@app.post("/predict_car")
def predict_item(car: Car) -> float:
    price_car = lr_model.predict(prepare_features(car))
    return price_car


@app.post("/predict_cars")
def predict_items(cars: List[Car]) -> List[float]:
    prices_cars = []
    for car in cars:
        prices_cars.append(lr_model.predict(prepare_features(car)))
    return prices_cars
