import ast

import openai
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import json
import os

from sklearn.ensemble import IsolationForest

load_dotenv(dotenv_path="D:\\smart_data_cleaner\\backend\\.env", override=True)
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
client = OpenAI(api_key=api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Уточни позже под фронт
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, encoding="utf-8")

    # Убираем inf, -inf, NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)

    # Приводим всё к строкам (временно, для MVP)
    df = df.astype(str)

    # Выдаём первые 5 строк
    preview_data = df.head(5).to_dict(orient="records")

    return {"preview": preview_data}


@app.post("/analyze/")
async def analyze_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, encoding="utf-8")
    df = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)

    # Анализ по колонкам
    analysis = []
    for col in df.columns:
        col_data = df[col]
        analysis.append({
            "column": col,
            "dtype": str(col_data.dtype),
            "nulls": int(col_data.isna().sum()),
            "unique": int(col_data.nunique()),
            "sample_values": col_data.dropna().astype(str).unique()[:3].tolist()
        })

    return {"columns": analysis}


@app.post("/outliers/")
async def detect_outliers(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, encoding="utf-8")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Оставляем только числовые колонки
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] == 0:
        return {"error": "Нет числовых данных для анализа выбросов."}

    response = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "developer",
                "content": "you will be given a list of columns of dataframe and head() of it. Look through the context"
                           " and state which columns will be the most useful to search for outliers. Give them as a "
                           "list: ['column_1', 'column_2', ...]. as an answer you should give only a list."
            },
            {
                "role": "user",
                "content": f"columns: {numeric_df.columns}, head: {numeric_df.head()}"
            }
        ]
    )
    print(response.output_text)

    # Модель Isolation Forest
    model = IsolationForest(n_estimators=100, max_samples=5000, contamination=0.05, n_jobs=-1, random_state=42)
    model.fit(numeric_df[eval(response.output_text)])

    # Предсказания: -1 = выброс, 1 = норм
    df["outlier"] = model.predict(numeric_df[ast.literal_eval(response.output_text)])

    # Выбрасываем нормальные строки, оставляем только выбросы
    outliers = df[df["outlier"] == -1].drop(columns=["outlier"])

    # Ограничим количество выбросов, чтобы не перегружать фронт
    outlier_preview = outliers.head(5).to_dict(orient="records")

    return {
        "outlier_count": len(outliers),
        "outlier_preview": outlier_preview,
        "columns_used": ast.literal_eval(response.output_text),
    }
