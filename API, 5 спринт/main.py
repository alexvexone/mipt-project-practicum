from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
import pandas as pd
import pickle
import io

with open("processed_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Проектный практикум: Прогнозирование одобрения кредита (Loan Approval Prediction)")

@app.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return JSONResponse(content={"error": "Только CSV файлы поддерживаются"}, status_code=400)

    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    # One-hot-encoding: получение дамми-признаков
    df = pd.get_dummies(df, drop_first=True)
    df = df.drop('loan_status', axis=1).fillna(df.mean())

    try:
        numeric = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
        scaler = StandardScaler()
        scaler.fit_transform(df[numeric])
        df[numeric] = scaler.transform(df[numeric])
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)