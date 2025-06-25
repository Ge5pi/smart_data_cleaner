import pandas as pd
import numpy as np
import os
import ast
import json

from sklearn.ensemble import IsolationForest
from celery_worker import celery_app
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="D:\\smart_data_cleaner\\backend\\.env", override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


@celery_app.task(name="tasks.upload_preview_task")
def upload_preview_task(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)
    df = df.astype(str)

    preview_data = df.head(5).to_dict(orient="records")
    return {"preview": preview_data}


@celery_app.task(name="tasks.analyze_file")
def analyze_file_task(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")
    df = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)

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

    return analysis


@celery_app.task(name="tasks.detect_outliers")
def detect_outliers_task(file_path, columns=None):
    df = pd.read_csv(file_path, encoding="utf-8")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if columns:
        selected_columns = json.loads(columns)
        numeric_df = df[selected_columns].select_dtypes(include=[np.number])
    else:
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
                           "list: [column_1, column_2, ...]. as an answer you should give only a list."
            },
            {
                "role": "user",
                "content": f"columns: {numeric_df.columns}, head: {numeric_df.head()}"
            }
        ]
    )

    columns_used = ast.literal_eval(response.output_text)

    model = IsolationForest(n_estimators=100, max_samples=5000, contamination=0.05, n_jobs=-1, random_state=42)
    model.fit(numeric_df[columns_used])

    df["outlier"] = model.predict(numeric_df[columns_used])
    outliers = df[df["outlier"] == -1].drop(columns=["outlier"])

    return {
        "outlier_count": len(outliers),
        "outlier_preview": outliers.head(5).to_dict(orient="records"),
        "columns_used": columns_used,
    }
