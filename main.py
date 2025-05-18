from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from pydantic import BaseModel, conlist
from pathlib import Path
import numpy as np


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Feature list
feature_names = [
    "Component Type (0=Coupler, 1=Motor)",
    "Running Hours Since Last Maintenance",
    "Ambient Temperature (°C)",
    "Vibration Level (mm/s)",
    "Motor Load (%)",
    "Coupler Torque (Nm)",
    "Bearing Temperature (°C)",
    "Lubrication Status (0=Low, 1=OK)",
    "Power Consumption (kW)",
    "Noise Level (dB)",
    "Phase Imbalance (%)",
    "Previous Breakdown Count",
    "Maintenance Frequency (days)",
    "Equipment Age (years)"
]

class FeatureInput(BaseModel):
    features: conlist(float, min_length= 14, max_length=14) # type: ignore
    
@app.get("/", response_class=HTMLResponse)
async def from_geet(request: Request):
    return templates.TemplateResponse("form.html", {"request": request,
                                                    "feature_names": feature_names,
                                                    "prediction": None})

@app.post("/predict")
async def predict(request: Request, features: list[float] = Form(...)):
    print(features)
    validated_input = FeatureInput(features=features)
    X = np.array([validated_input.features])
    print(X)

    return templates.TemplateResponse("form.html", {
            "request": request,
            "feature_names": feature_names,
            "features": validated_input.features,
            "prediction": 1000
        })
