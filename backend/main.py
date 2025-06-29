import uuid
from typing import Optional

import openai
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import json
from pathlib import Path
import os
from sklearn.ensemble import IsolationForest
import httpx
from sklearn.preprocessing import LabelEncoder
from starlette.responses import FileResponse
from tabpfn import TabPFNClassifier, TabPFNRegressor
import pinecone
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm

load_dotenv(dotenv_path="D:\\smart_data_cleaner\\backend\\.env", override=True)

api_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
webhook_test_key = os.getenv("WEBHOOK_TEST_KEY")

client = OpenAI(api_key=api_key)
pc = pinecone.Pinecone(api_key=pinecone_key)
app = FastAPI()

N8N_WEBHOOK_URL = f"http://localhost:5678/webhook-test/{webhook_test_key}"
index_name = "soda-index"
index = pc.Index(index_name)
TEXT_COLUMN = "text"
ID_COLUMN = "id"
BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-3-small"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path("temp_cleaned_files")
TEMP_DIR.mkdir(exist_ok=True)


async def trigger_n8n_workflow(file_id: str):
    if not N8N_WEBHOOK_URL:
        print("Переменная N8N_WEBHOOK_URL не установлена. Пропуск вызова вебхука.")
        return

    payload = {"file_id": file_id}

    try:
        async with httpx.AsyncClient() as web_client:
            print(f"Отправка вебхука в n8n для file_id: {file_id}")
            response = await web_client.post(N8N_WEBHOOK_URL, json=payload, timeout=10)
            response.raise_for_status()
            print(f"Вебхук успешно отправлен. Статус ответа n8n: {response.status_code}")
    except httpx.RequestError as e:
        print(f"Ошибка при вызове вебхука n8n: {e}")


def impute_missing_values_with_tabpfn(df, target_column, max_samples=1000):
    df_work = df.copy()

    if not df_work[target_column].isna().any():
        return df_work[target_column]

    mask_missing = df_work[target_column].isna()
    df_complete = df_work[~mask_missing]
    df_missing = df_work[mask_missing]

    if len(df_complete) == 0:
        return df_work[target_column]

    if len(df_complete) > max_samples:
        df_complete = df_complete.sample(n=max_samples, random_state=42)

    feature_columns = [col for col in df_work.columns if col != target_column]
    useful_features = []
    label_encoders = {}

    for col in feature_columns:
        if df_complete[col].isna().sum() / len(df_complete) > 0.5:
            continue

        if df_complete[col].nunique() > 100:
            continue

        useful_features.append(col)

    if len(useful_features) == 0:
        return df_work[target_column]

    X_train = df_complete[useful_features].copy()
    y_train = df_complete[target_column].copy()
    X_predict = df_missing[useful_features].copy()

    for col in useful_features:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()

            X_train[col] = X_train[col].fillna('missing')
            X_predict[col] = X_predict[col].fillna('missing')

            all_values = list(set(X_train[col].tolist() + X_predict[col].tolist()))
            le.fit(all_values)

            X_train[col] = le.transform(X_train[col])
            X_predict[col] = le.transform(X_predict[col])

            label_encoders[col] = le

    for col in useful_features:
        if X_train[col].dtype in ['int64', 'float64']:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_predict[col] = X_predict[col].fillna(median_val)

    try:
        if y_train.dtype == 'object' or y_train.nunique() < 20:
            le_target = LabelEncoder()
            y_train_encoded = le_target.fit_transform(y_train.astype(str))

            model = TabPFNClassifier(device='cpu')
            model.fit(X_train.values, y_train_encoded)

            predictions_encoded = model.predict(X_predict.values)
            predictions = le_target.inverse_transform(predictions_encoded)

            if y_train.dtype != 'object':
                predictions = pd.Series(predictions).astype(y_train.dtype).values

        else:
            model = TabPFNRegressor(device='cpu')
            model.fit(X_train.values, y_train.values)

            predictions = model.predict(X_predict.values)

            predictions = pd.Series(predictions).astype(y_train.dtype).values

        result = df_work[target_column].copy()
        result.loc[mask_missing] = predictions

        return result

    except Exception as e:
        print(f"Ошибка при использовании TabPFN для столбца {target_column}: {e}")
        return df_work[target_column]


def combine_columns(row):
    return " | ".join([f"{col}: {str(row[col])}" for col in row.index])


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(6))
def get_embeddings(texts: list[str]):
    client = openai.OpenAI(api_key=api_key)

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [r["embedding"] for r in response["data"]]

# MODIFIED: Only this endpoint now accepts a file upload.
@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, encoding="utf-8")
    file_id = str(uuid.uuid4())

    df = df.replace([np.inf, -np.inf], np.nan)

    output_filename = f"{file_id}.csv"
    output_path = TEMP_DIR / output_filename
    df.to_csv(output_path, index=False)


    df = df.where(pd.notnull(df), None)

    df = df.astype(str)
    preview_data = df.to_dict(orient="records")
    if ID_COLUMN not in df.columns:
        df[ID_COLUMN] = [f"row-{i}" for i in range(len(df))]

    df[TEXT_COLUMN] = df.apply(combine_columns, axis=1)

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Uploading to Pinecone"):
        batch_df = df.iloc[i:i + BATCH_SIZE]
        texts = batch_df[TEXT_COLUMN].astype(str).tolist()
        ids = batch_df[ID_COLUMN].astype(str).tolist()


        embeddings = get_embeddings(texts)

        print(embeddings)

        vectors = [
            {
                "id": ids[j],
                "values": embeddings[j],
                "metadata": {
                    "original_text": texts[j],
                    "file_id": file_id
                }
            } for j in range(len(ids))
        ]

        print("Vectors:", vectors)

        index.upsert(vectors)
    return {"preview": preview_data, "file_id": file_id}


# MODIFIED: Accepts file_id instead of the file.
@app.post("/analyze/")
async def analyze_csv(file_id: str = Form(...)):
    file_path = TEMP_DIR / f"{file_id}.csv"
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    df = pd.read_csv(file_path, encoding="utf-8")
    df = df.replace([np.inf, -np.inf], np.nan)

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


# MODIFIED: Accepts file_id instead of the file.
@app.post("/impute-missing/")
async def impute_missing_values(file_id: str = Form(...), columns: Optional[str] = Form(None)):
    file_path = TEMP_DIR / f"{file_id}.csv"
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    df = pd.read_csv(file_path, encoding="utf-8")
    df = df.replace([np.inf, -np.inf], np.nan)

    if columns:
        selected_columns = json.loads(columns)
    else:
        selected_columns = [col for col in df.columns if df[col].isna().any()]

    if not selected_columns:
        return {"error": "Нет столбцов с пропущенными значениями для обработки."}

    missing_before = {col: int(df[col].isna().sum()) for col in selected_columns}

    df_imputed = df.copy()
    processing_results = {}

    for col in selected_columns:
        if df[col].isna().any():
            try:
                print(f"Обрабатываем столбец: {col}")
                df_imputed[col] = impute_missing_values_with_tabpfn(df, col)
                df_imputed[col] = df_imputed[col].replace([np.inf, -np.inf], np.nan)
                processing_results[col] = "success"
            except Exception as e:
                print(f"Ошибка при обработке столбца {col}: {e}")
                processing_results[col] = f"error: {str(e)}"

    missing_after = {col: int(df_imputed[col].isna().sum()) for col in selected_columns}

    df_imputed = df_imputed.replace([np.inf, -np.inf], np.nan)
    preview_data = df_imputed.fillna("null").to_dict(orient="records")

    return {
        "preview": preview_data,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "processing_results": processing_results,
        "total_rows": len(df_imputed)
    }


# MODIFIED: Accepts file_id instead of the file.
@app.post("/outliers/")
async def detect_outliers(file_id: str = Form(...), columns: Optional[str] = Form(None)):
    file_path = TEMP_DIR / f"{file_id}.csv"
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    df = pd.read_csv(file_path, encoding="utf-8")

    if columns:
        import json
        selected_columns = json.loads(columns)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=selected_columns)
        numeric_df = df[selected_columns].select_dtypes(include=[np.number])
    else:
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_df = df.select_dtypes(include=[np.number])
        df = df.dropna(subset=numeric_df.columns.tolist())

    if numeric_df.shape[1] == 0:
        return {"error": "Нет числовых данных для анализа выбросов."}

    model = IsolationForest(n_estimators=100, max_samples=5000, contamination=0.05, n_jobs=-1, random_state=42)
    model.fit(numeric_df)

    df["outlier"] = model.predict(numeric_df)
    outliers = df[df["outlier"] == -1].drop(columns=["outlier"])
    outlier_preview = outliers.head(5).replace([np.inf, -np.inf, np.nan], None).to_dict(orient="records")

    return {
        "outlier_count": len(outliers),
        "outlier_preview": outlier_preview,
        "columns_used": selected_columns,
    }


# MODIFIED: This is the main modification endpoint.
# It now accepts file_id, loads the file, processes it, and saves it back to the same path.
@app.post("/clean-data/")
async def clean_data(file_id: str = Form(...), impute_columns: Optional[str] = Form(None)):
    file_path = TEMP_DIR / f"{file_id}.csv"
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    df = pd.read_csv(file_path)
    df = df.replace([np.inf, -np.inf], np.nan)

    if impute_columns:
        impute_cols = json.loads(impute_columns)
        for col in impute_cols:
            if df[col].isna().any():
                df[col] = impute_missing_values_with_tabpfn(df, col)

    # Overwrite the existing file with the cleaned data
    df.to_csv(file_path, index=False)

    return {
        "message": "File cleaned successfully",
        "file_id": file_id,  # Return the same file_id
        "new_data": df.fillna("null").to_dict(orient="records")
    }


# MODIFIED: Renamed function for clarity. The endpoint remains the same.
@app.get("/download-cleaned/{file_id}")
async def download_cleaned_file(file_id: str):
    file_path = TEMP_DIR / f"{file_id}.csv"

    if not file_path.is_file() or not str(file_path.resolve()).startswith(str(TEMP_DIR.resolve())):
        raise HTTPException(status_code=404, detail="File not found or access denied.")

    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename="cleaned_data.csv",
        headers={"Content-Disposition": "attachment; filename=cleaned_data.csv"}
    )
