from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json

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
