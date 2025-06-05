from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
import pandas as pd
import pickle
import io
import numpy as np

# Загрузка моделей
with open("processed_model.pkl", "rb") as f:
    processed_model = pickle.load(f)

with open("processed_model_cohort.pkl", "rb") as f:
    processed_model_cohort = pickle.load(f)

app = FastAPI(title="Проектный практикум: Прогнозирование одобрения кредита (Loan Approval Prediction)")

@app.post("/predict/file")
async def predict_from_file(
    file: UploadFile = File(...),
    model_type: str = Query("default", enum=["default", "cohort"], description="Выбор модели: 'default' или 'cohort'")
):
    if not file.filename.endswith(".csv"):
        return JSONResponse(content={"error": "Только CSV файлы поддерживаются"}, status_code=400)

    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    
    if model_type != 'default':
        # Создание когорт
        df['risk_cohort'] = np.select(
            [
                (df['loan_grade'].isin(['A', 'B'])) & (df['loan_percent_income'] < 0.2) &
                (df['cb_person_default_on_file'] == 'N'),
                (df['loan_grade'].isin(['C', 'D'])) & (df['loan_percent_income'].between(0.2, 0.4)),
                (df['loan_grade'].isin(['E', 'F', 'G'])) | (df['loan_percent_income'] > 0.4) |
                (df['cb_person_default_on_file'] == 'Y')
            ],
            ['Low Risk', 'Medium Risk', 'High Risk'],
            default='Medium Risk'
        )

        # Удаление id
        df = df.drop(columns=['id'])
        bins = [0, 25, 35, 50, float('inf')]
        labels = ['<25', '25-35', '36-50', '>50']
        df['age_group'] = pd.cut(df['person_age'], bins=bins, labels=labels, right=False)
        
        # Обработка выбросов (метод IQR)
        def cap_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            return df

        numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
                        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
        for col in numeric_cols:
            df = cap_outliers(df, col)

        # Обработка пропусков
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file', 'risk_cohort']
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        # Преобразование cb_person_default_on_file в бинарный признак
        df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})


    try:
        # Предобработка
        df = pd.get_dummies(df, drop_first=True)
        df = df.drop('loan_status', axis=1).fillna(df.mean())

        numeric = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
        scaler = StandardScaler()
        df[numeric] = scaler.fit_transform(df[numeric])

        # Выбор модели
        model = processed_model if model_type == "default" else processed_model_cohort

        # Предсказание
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)