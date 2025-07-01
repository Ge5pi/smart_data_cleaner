import uuid
from typing import Optional
import tiktoken
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

index_name = "soda-index"
index = pc.Index(index_name)
TEXT_COLUMN = "text"
ID_COLUMN = "id"
BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-3-small"
AGENT_MODEL = "gpt-4o"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path("temp_cleaned_files")
TEMP_DIR.mkdir(exist_ok=True)


def format_row(row_index: int, row: pd.Series) -> str:
    """Форматирует строку DataFrame в текстовое представление."""
    return f"Row-{row_index + 1}: " + " | ".join([f"{col}: {row[col]}" for col in row.index])


MAX_TOKENS_PER_CHUNK = 8192
MAX_CHUNKS = 999

encoding = tiktoken.encoding_for_model("text-embedding-3-small")
file_metadata_storage = {}
session_cache = {}

SYSTEM_PROMPT = """
Ты — элитный AI-аналитик данных. Ты работаешь в интерактивной сессии с пользователем.
1.  У тебя есть DataFrame, хранящийся в переменной `df`. Все твои операции должны изменять этот DataFrame.
2.  Твой главный инструмент — `execute_python_code`. Используй его для выполнения любых операций с `df`: фильтрации, создания новых столбцов, группировки (`groupby`), агрегации (`agg`).
3.  Для неструктурированных вопросов, требующих смыслового поиска по тексту, используй `answer_question_from_context`.
4.  Думай по шагам. Если нужно, вызови несколько инструментов подряд.
5.  После выполнения кода покажи пользователю результат и кратко объясни, что ты сделал.
6.  Всегда будь готов к следующему вопросу, который будет основан на текущем состоянии данных.
"""


def num_tokens(text: str) -> int:
    return len(encoding.encode(text))


def impute_missing_values_with_tabpfn(df, target_column, max_samples=1000):
    """Заполняет пропуски с помощью TabPFN (оригинальная функция)."""
    df_work = df.copy()
    if not df_work[target_column].isna().any(): return df_work[target_column]
    mask_missing = df_work[target_column].isna()
    df_complete = df_work[~mask_missing]
    df_missing = df_work[mask_missing]
    if len(df_complete) == 0: return df_work[target_column]
    if len(df_complete) > max_samples:
        df_complete = df_complete.sample(n=max_samples, random_state=42)
    feature_columns = [col for col in df_work.columns if col != target_column]
    useful_features, label_encoders = [], {}
    for col in feature_columns:
        if df_complete[col].isna().sum() / len(df_complete) > 0.5 or df_complete[col].nunique() > 100:
            continue
        useful_features.append(col)
    if not useful_features: return df_work[target_column]
    X_train, y_train, X_predict = df_complete[useful_features].copy(), df_complete[target_column].copy(), df_missing[
        useful_features].copy()
    for col in useful_features:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            X_train[col], X_predict[col] = X_train[col].fillna('missing'), X_predict[col].fillna('missing')
            all_values = list(set(X_train[col].tolist() + X_predict[col].tolist()))
            le.fit(all_values)
            X_train[col], X_predict[col] = le.transform(X_train[col]), le.transform(X_predict[col])
            label_encoders[col] = le
        elif X_train[col].dtype in ['int64', 'float64']:
            median_val = X_train[col].median()
            X_train[col], X_predict[col] = X_train[col].fillna(median_val), X_predict[col].fillna(median_val)
    try:
        if y_train.dtype == 'object' or y_train.nunique() < 20:
            le_target = LabelEncoder()
            y_train_encoded = le_target.fit_transform(y_train.astype(str))
            model = TabPFNClassifier(device='cpu')
            model.fit(X_train.values, y_train_encoded)
            predictions_encoded = model.predict(X_predict.values)
            predictions = le_target.inverse_transform(predictions_encoded)
            if y_train.dtype != 'object': predictions = pd.Series(predictions).astype(y_train.dtype).values
        else:
            model = TabPFNRegressor(device='cpu')
            model.fit(X_train.values, y_train.values)
            predictions = model.predict(X_predict.values)
            predictions = pd.Series(predictions).astype(y_train.dtype).values
        result = df_work[target_column].copy()
        result.loc[mask_missing] = predictions
        return result
    except Exception:
        return df_work[target_column]


def execute_python_code(session_id: str, code: str) -> str:
    """
    Выполняет строку Python-кода в контексте текущего DataFrame сессии.
    Улучшенная версия: умеет захватывать результат последней строки кода.
    """
    print(f"TOOL (session: {session_id}): Выполнение кода:\n---\n{code}\n---")
    if session_id not in session_cache:
        return "Ошибка: Сессия не найдена. Начните новую сессию."

    current_df = session_cache[session_id]["dataframe"]
    local_scope = {"df": current_df, "pd": pd}

    try:
        # Разделяем код на отдельные инструкции
        lines = code.strip().split('\n')
        # Все строки, кроме последней, выполняем через exec
        exec_lines = lines[:-1]
        # Последнюю строку пытаемся вычислить через eval, чтобы получить результат
        eval_line = lines[-1]

        if exec_lines:
            exec("\n".join(exec_lines), {"pd": pd}, local_scope)

        # Вычисляем результат последней строки
        output = eval(eval_line, {"pd": pd}, local_scope)

        # Обновляем DataFrame в кэше, если он был изменен в процессе
        session_cache[session_id]["dataframe"] = local_scope['df']

        # Если результат - это DataFrame, вернем его часть
        if isinstance(output, pd.DataFrame):
            return output.head(15).to_markdown()

        # В противном случае просто вернем результат (например, число)
        return str(output)

    except Exception as e:
        # Если eval не сработал (например, последняя строка - это `df = ...`),
        # пробуем выполнить весь код через exec и вернуть стандартный head()
        try:
            exec(code, {"pd": pd}, local_scope)
            session_cache[session_id]["dataframe"] = local_scope['df']
            return local_scope['df'].head(15).to_markdown()
        except Exception as exec_e:
            return f"Ошибка выполнения кода: {exec_e}"


def run_rag_pipeline(file_id: str, query: str) -> str:
    """Выполняет RAG-поиск по исходному файлу."""
    query_embedding = get_embeddings([query])[0]
    search_results = index.query(
        vector=query_embedding, top_k=7, filter={"file_id": file_id}, include_metadata=True
    )
    context = ""
    if search_results['matches']:
        for match in search_results['matches']:
            context += match['metadata']['original_text'] + "\n---\n"
    if not context: return "Не удалось найти релевантную информацию в файле."
    rag_messages = [
        {"role": "system",
         "content": "Ответь на вопрос пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном контексте."},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"}
    ]
    response = client.chat.completions.create(model=AGENT_MODEL, messages=rag_messages, temperature=0.0)
    return response.choices[0].message.content


def answer_question_from_context(session_id: str, query: str) -> str:
    """Инструмент-обертка для RAG, использующий ID сессии."""
    print(f"TOOL (session: {session_id}): RAG-запрос: '{query}'")
    file_id = session_cache.get(session_id, {}).get("file_id")
    if not file_id: return "Ошибка: не найден исходный file_id для RAG-поиска."
    return run_rag_pipeline(file_id, query)


tools_definition = [
    {
        "type": "function", "function": {
            "name": "execute_python_code",
            "description": "Выполняет Python-код с pandas для манипуляции данными (фильтрация, сортировка, группировка, агрегация). Оперируй с переменной 'df'.",
            "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}, "code": {"type": "string", "description": "Код на Python для выполнения."}}, "required": ["session_id", "code"]},
        }
    },
    {
        "type": "function", "function": {
            "name": "answer_question_from_context",
            "description": "Ответить на нечеткий, семантический вопрос, требующий поиска по смыслу в исходном документе. Используй, если анализ данных с pandas не подходит.",
            "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}, "query": {"type": "string", "description": "Оригинальный вопрос."}}, "required": ["session_id", "query"]},
        }
    }
]


available_functions = {
    "execute_python_code": execute_python_code,
    "answer_question_from_context": answer_question_from_context,
}


def combine_columns(row):
    return " | ".join([f"{col}: {str(row[col])}" for col in row.index])


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(6))
def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Получает эмбеддинги для списка текстов."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [r.embedding for r in response.data]


@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):
    """Загружает CSV, сохраняет его и индексирует для RAG."""
    file_id = str(uuid.uuid4())
    output_path = TEMP_DIR / f"{file_id}.csv"

    df = pd.read_csv(file.file, encoding="utf-8")
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv(output_path, index=False)

    file_metadata_storage[file_id] = {
        "total_rows": len(df),
        "column_names": df.columns.tolist(),
        "file_path": str(output_path)
    }

    # Индексация для RAG
    vectors_to_upsert = []
    df_rag = df.where(pd.notnull(df), 'null')
    for i in tqdm(range(len(df_rag)), desc=f"Подготовка векторов для {file_id}"):
        row_text = format_row(i, df_rag.iloc[i])
        vectors_to_upsert.append({
            "id": f"{file_id}-row-{i}", "values": [],
            "metadata": {"file_id": file_id, "original_text": row_text}
        })

    for i in tqdm(range(0, len(vectors_to_upsert), BATCH_SIZE), desc=f"Индексация в Pinecone для {file_id}"):
        batch = vectors_to_upsert[i:i + BATCH_SIZE]
        texts_to_embed = [v['metadata']['original_text'] for v in batch]
        try:
            embeddings = get_embeddings(texts_to_embed)
            for j, vector in enumerate(batch): vector["values"] = embeddings[j]
            index.upsert(vectors=batch)
        except Exception as e:
            print(f"Ошибка при обработке пачки {i}-{i + BATCH_SIZE}: {e}")

    preview_data = df.head(10).fillna("null").to_dict(orient="records")
    return {"preview": preview_data, "file_id": file_id}



@app.post("/sessions/start")
async def start_session(file_id: str = Form(...)):
    """Начинает новую аналитическую сессию."""
    if file_id not in file_metadata_storage:
        raise HTTPException(status_code=404, detail="Файл с таким ID не найден.")
    session_id = str(uuid.uuid4())
    df = pd.read_csv(file_metadata_storage[file_id]['file_path'])
    session_cache[session_id] = {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "dataframe": df, "file_id": file_id
    }
    initial_schema = df.info(verbose=False)
    return {
        "session_id": session_id,
        "message": "Сессия успешно начата.",
        "schema": str(initial_schema)
    }


@app.post("/sessions/ask")
async def session_ask(session_id: str = Form(...), query: str = Form(...)):
    """Задает вопрос в рамках существующей сессии."""
    if session_id not in session_cache:
        raise HTTPException(status_code=404, detail="Сессия не найдена.")
    messages = session_cache[session_id]["messages"]
    messages.append({"role": "user", "content": query})
    try:
        for _ in range(7):
            response = client.chat.completions.create(
                model=AGENT_MODEL, messages=messages, tools=tools_definition, tool_choice="auto"
            )
            response_message = response.choices[0].message
            if response_message.tool_calls:
                messages.append(response_message)
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    function_args['session_id'] = session_id
                    function_response = function_to_call(**function_args)
                    messages.append({
                        "tool_call_id": tool_call.id, "role": "tool",
                        "name": function_name, "content": function_response
                    })
            else:
                messages.append(response_message)
                session_cache[session_id]["messages"] = messages
                return {"answer": response_message.content}
        return {"answer": "Не удалось получить ответ за максимальное количество шагов."}
    except Exception as e:
        session_cache[session_id]["messages"] = messages
        raise HTTPException(status_code=500, detail=f"Произошла внутренняя ошибка: {str(e)}")


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
