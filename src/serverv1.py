

import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from os import walk


app = FastAPI()

class Balance(BaseModel):
    ODI: float
    PT: float
    SS: float
    PI: float
    LL: float
    
@app.post('/predict')
def predict(balance: Balance):
    models = get_models()
    features = pd.DataFrame(
            zip(
                [balance.ODI],
                [balance.PT],
                [balance.SS],
                [balance.PI],
                [balance.LL]
                ),
            columns=['ODI', 'PT', 'SS', 'PI', 'LL'])
    prediction = np.mean([model.predict(features) for model in models], 0)
    return {
        "prediction": prediction[0]
    }

def get_models():
    mdl_lbl = [];
    for _, _, filename in walk('../models/v1/'):
        mdl_lbl.extend(filename);

    models = [];
    for filename in mdl_lbl:
        models.append(joblib.load('../models/v1/' + f"{filename}"));

    return models
        


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
