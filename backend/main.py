import uuid
from typing import Optional
import json
from pathlib import Path
import os
import io

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

import openai
import pinecone
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier, TabPFNRegressor

# ==============================================================================
# 1. КОНФИГУРАЦИЯ И ИНИЦИАЛИЗАЦИЯ
# ==============================================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

client = openai.OpenAI(api_key=api_key)
pc = pinecone.Pinecone(api_key=pinecone_key)
app = FastAPI()

# Настройки моделей и констант
EMBEDDING_MODEL = "text-embedding-3-small"
AGENT_MODEL = "gpt-4o"
CRITIC_MODEL = "gpt-4o"
INDEX_NAME = "soda-index"
BATCH_SIZE = 100

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Создание директории для временных файлов
TEMP_DIR = Path("temp_cleaned_files")
TEMP_DIR.mkdir(exist_ok=True)

# Инициализация векторной базы данных Pinecone
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=1536, metric='cosine')
index = pc.Index(INDEX_NAME)

# Хранилища состояний в памяти
file_metadata_storage = {}
session_cache = {}


# ==============================================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И ИНСТРУМЕНТЫ ИЗ ОРИГИНАЛЬНОГО ФАЙЛА
# ==============================================================================

def format_row(row_index: int, row: pd.Series) -> str:
    return f"Row-{row_index + 1}: " + " | ".join([f"{col}: {row[col]}" for col in row.index])


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(6))
def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [r.embedding for r in response.data]


def impute_missing_values_with_tabpfn(df, target_column, max_samples=1000):
    # ... (Ваш код функции impute_missing_values_with_tabpfn без изменений)
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


# ==============================================================================
# 3. НОВЫЙ МОДУЛЬ ОЦЕНКИ И УЛУЧШЕНИЯ ОТВЕТА
# ==============================================================================

def get_critic_evaluation(query: str, answer: str) -> dict:
    """Вызывает модель-критика для оценки ответа."""
    critic_prompt = f"""
    You are a meticulous and impartial AI critic. Your task is to evaluate a given answer based on a user's query.
    Provide your evaluation in a structured JSON format.

    **User Query:**
    {query}

    **Answer to Evaluate:**
    {answer}

    **Evaluation Criteria:**
    1.  **relevance**: How relevant is the answer to the user's query? (1-5, where 5 is most relevant)
    2.  **completeness**: Does the answer fully address all parts of the query? (1-5, where 5 is most complete)
    3.  **confidence**: How confident are you in the factual accuracy of the answer, based on the likely context of a pandas DataFrame? (1-5, where 5 is most confident)
    4.  **feedback**: Provide brief, constructive feedback for improvement if the scores are low. If the answer is good, say "The answer is sufficient.".

    **Output (JSON format only):**
    """
    try:
        response = client.chat.completions.create(
            model=CRITIC_MODEL,
            messages=[{"role": "user", "content": critic_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in critic model: {e}")
        return {"relevance": 5, "completeness": 5, "confidence": 5,
                "feedback": "Critic model failed, assuming success."}


def get_refined_answer(history: list, original_answer: str, feedback: str) -> str:
    """Вызывает модель-улучшателя для исправления ответа."""
    refiner_prompt = f"""
    You are a world-class data analyst. Your task is to refine and improve a previous answer based on the feedback provided by a critic.
    The last user question is at the end of the conversation history.

    **Original Conversation History:**
    {json.dumps(history, indent=2, ensure_ascii=False)}

    **Your previous (unsatisfactory) answer:**
    {original_answer}

    **Critic's Feedback for Improvement:**
    {feedback}

    **Your Task:**
    Generate a new, improved, and complete answer that directly addresses the user's last question and the critic's feedback.
    """
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[{"role": "user", "content": refiner_prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content


# ==============================================================================
# 4. ЯДРО AI-АГЕНТА (Инструменты и определение)
# ==============================================================================

SYSTEM_PROMPT = """
Ты — элитный AI-аналитик данных. Ты работаешь в интерактивной сессии с пользователем.
1.  У тебя есть DataFrame, хранящийся в переменной `df`. Все твои операции должны изменять этот DataFrame.
2.  Твой главный инструмент — `execute_python_code`. Используй его для выполнения любых операций с `df`: фильтрации, создания новых столбцов, группировки (`groupby`), агрегации (`agg`).
3.  Для неструктурированных вопросов, требующих смыслового поиска по тексту, используй `answer_question_from_context`.
4.  Думай по шагам. Если нужно, вызови несколько инструментов подряд.
5.  После выполнения кода покажи пользователю результат и кратко объясни, что ты сделал.
6.  Всегда будь готов к следующему вопросу, который будет основан на текущем состоянии данных.
"""


def execute_python_code(session_id: str, code: str) -> str:
    """Выполняет Python-код, умеет захватывать результат последней строки."""
    print(f"TOOL (session: {session_id}): Выполнение кода:\n---\n{code}\n---")
    if session_id not in session_cache: return "Ошибка: Сессия не найдена."
    current_df = session_cache[session_id]["dataframe"]
    local_scope = {"df": current_df, "pd": pd, "impute_missing_values_with_tabpfn": impute_missing_values_with_tabpfn,
                   "IsolationForest": IsolationForest}
    try:
        lines = code.strip().split('\n')
        exec_lines, eval_line = lines[:-1], lines[-1]
        if exec_lines:
            exec("\n".join(exec_lines), {"pd": pd}, local_scope)
        output = eval(eval_line, {"pd": pd, "impute_missing_values_with_tabpfn": impute_missing_values_with_tabpfn,
                                  "IsolationForest": IsolationForest}, local_scope)
        session_cache[session_id]["dataframe"] = local_scope['df']
        if isinstance(output, pd.DataFrame):
            return output.head(15).to_markdown()
        return str(output)
    except Exception:
        try:
            exec(code, {"pd": pd}, local_scope)
            session_cache[session_id]["dataframe"] = local_scope['df']
            return local_scope['df'].head(15).to_markdown()
        except Exception as exec_e:
            return f"Ошибка выполнения кода: {exec_e}"


def run_rag_pipeline(file_id: str, query: str) -> str:
    """Выполняет RAG-поиск по исходному файлу."""
    query_embedding = get_embeddings([query])[0]
    search_results = index.query(vector=query_embedding, top_k=7, filter={"file_id": file_id}, include_metadata=True)
    context = ""
    if search_results.get('matches'):
        for match in search_results['matches']:
            context += match['metadata']['original_text'] + "\n---\n"
    if not context: return "Не удалось найти релевантную информацию в файле."
    rag_messages = [{"role": "system",
                     "content": "Ответь на вопрос пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном контексте."},
                    {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"}]
    response = client.chat.completions.create(model=AGENT_MODEL, messages=rag_messages, temperature=0.0)
    return response.choices[0].message.content


def answer_question_from_context(session_id: str, query: str) -> str:
    """Инструмент-обертка для RAG."""
    print(f"TOOL (session: {session_id}): RAG-запрос: '{query}'")
    file_id = session_cache.get(session_id, {}).get("file_id")
    if not file_id: return "Ошибка: не найден исходный file_id."
    return run_rag_pipeline(file_id, query)


tools_definition = [{"type": "function", "function": {"name": "execute_python_code",
                                                      "description": "Выполняет Python-код с pandas для манипуляции данными. Оперируй с переменной 'df'.",
                                                      "parameters": {"type": "object",
                                                                     "properties": {"session_id": {"type": "string"},
                                                                                    "code": {"type": "string"}},
                                                                     "required": ["session_id", "code"]}}},
                    {"type": "function", "function": {"name": "answer_question_from_context",
                                                      "description": "Ответить на нечеткий, семантический вопрос.",
                                                      "parameters": {"type": "object",
                                                                     "properties": {"session_id": {"type": "string"},
                                                                                    "query": {"type": "string"}},
                                                                     "required": ["session_id", "query"]}}}]
available_functions = {"execute_python_code": execute_python_code,
                       "answer_question_from_context": answer_question_from_context}


# ==============================================================================
# 5. ЭНДПОИНТЫ FastAPI
# ==============================================================================

@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):
    """Загружает CSV, сохраняет его и индексирует для RAG."""
    # ... (Ваш код эндпоинта /upload/ без изменений)
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
    # ... (Ваш код эндпоинта /sessions/start без изменений)
    if file_id not in file_metadata_storage:
        raise HTTPException(status_code=404, detail="Файл с таким ID не найден.")
    session_id = str(uuid.uuid4())
    df = pd.read_csv(file_metadata_storage[file_id]['file_path'])
    session_cache[session_id] = {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "dataframe": df, "file_id": file_id
    }
    return {"session_id": session_id, "message": "Сессия успешно начата."}


@app.post("/sessions/ask")
async def session_ask(session_id: str = Form(...), query: str = Form(...)):
    """Задает вопрос в рамках сессии с новым циклом оценки и улучшения."""
    if session_id not in session_cache:
        raise HTTPException(status_code=404, detail="Сессия не найдена.")

    # --- НОВОЕ: Добавление контекста о DataFrame к каждому запросу ---
    df = session_cache[session_id]["dataframe"]
    buf = io.StringIO()
    df.info(buf=buf)
    df_info = buf.getvalue()
    df_head = df.head().to_markdown()

    contextual_query = f"""
Контекст данных:
1. Схема данных (df.info()): {df_info}
2. Первые 5 строк (df.head()): {df_head}
---
Вопрос пользователя: {query}
"""
    messages = session_cache[session_id]["messages"]
    messages.append({"role": "user", "content": contextual_query})

    try:
        # --- ШАГ 1: ПОЛУЧЕНИЕ ПЕРВОНАЧАЛЬНОГО ОТВЕТА АГЕНТА ---
        initial_answer = ""
        for _ in range(5):  # Агентский цикл
            response = client.chat.completions.create(model=AGENT_MODEL, messages=messages, tools=tools_definition,
                                                      tool_choice="auto")
            response_message = response.choices[0].message
            if response_message.tool_calls:
                messages.append(response_message)
                for tool_call in response_message.tool_calls:
                    function_name, function_args = tool_call.function.name, json.loads(tool_call.function.arguments)
                    function_args['session_id'] = session_id
                    function_response = available_functions[function_name](**function_args)
                    messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name,
                                     "content": function_response})
            else:
                initial_answer = response_message.content
                messages.append(response_message)
                break

        if not initial_answer:
            raise HTTPException(status_code=500, detail="Модель не смогла сгенерировать первоначальный ответ.")

        # --- ШАГ 2: ОЦЕНКА ОТВЕТА КРИТИКОМ ---
        evaluation = get_critic_evaluation(query, initial_answer)
        print(f"Critic Evaluation: {evaluation}")

        final_answer = initial_answer
        # --- ШАГ 3: УЛУЧШЕНИЕ ОТВЕТА, ЕСЛИ НЕОБХОДИМО ---
        if evaluation.get("confidence", 5) < 4:
            print("Ответ неудовлетворительный. Запускается модель-улучшатель...")
            final_answer = get_refined_answer(messages, initial_answer, evaluation.get("feedback"))
            messages.append({"role": "assistant", "content": final_answer})

        session_cache[session_id]["messages"] = messages
        return {"answer": final_answer, "evaluation": evaluation}

    except Exception as e:
        session_cache[session_id]["messages"] = messages
        raise HTTPException(status_code=500, detail=f"Произошла внутренняя ошибка: {str(e)}")


# --- Ваши оригинальные эндпоинты для очистки ---

@app.post("/analyze/")
async def analyze_csv(file_id: str = Form(...)):
    # ... (Ваш код эндпоинта /analyze/ без изменений)
    if file_id not in file_metadata_storage: raise HTTPException(status_code=404, detail="File not found.")
    df = pd.read_csv(file_metadata_storage[file_id]['file_path'])
    analysis = [{"column": col, "dtype": str(df[col].dtype), "nulls": int(df[col].isna().sum()),
                 "unique": int(df[col].nunique()), "sample_values": df[col].dropna().astype(str).unique()[:3].tolist()}
                for col in df.columns]
    return {"columns": analysis}


@app.post("/impute-missing/")
async def impute_missing_values(file_id: str = Form(...), columns: Optional[str] = Form(None)):
    # ... (Ваш код эндпоинта /impute-missing/ без изменений)
    if file_id not in file_metadata_storage: raise HTTPException(status_code=404, detail="File not found.")
    df = pd.read_csv(file_metadata_storage[file_id]['file_path'])
    selected_columns = json.loads(columns) if columns else [col for col in df.columns if df[col].isna().any()]
    if not selected_columns: return {"error": "Нет столбцов с пропущенными значениями."}
    missing_before = {col: int(df[col].isna().sum()) for col in selected_columns}
    df_imputed = df.copy()
    for col in selected_columns:
        if df[col].isna().any():
            df_imputed[col] = impute_missing_values_with_tabpfn(df, col)
    df_imputed.to_csv(file_metadata_storage[file_id]['file_path'], index=False)  # Сохраняем изменения
    return {"message": "Заполнение пропусков завершено."}


@app.post("/outliers/")
async def detect_outliers(file_id: str = Form(...), columns: Optional[str] = Form(None)):
    # ... (Ваш код эндпоинта /outliers/ без изменений)
    if file_id not in file_metadata_storage: raise HTTPException(status_code=404, detail="File not found.")
    df = pd.read_csv(file_metadata_storage[file_id]['file_path'])
    selected_columns = json.loads(columns) if columns else df.select_dtypes(include=np.number).columns.tolist()
    numeric_df = df[selected_columns].select_dtypes(include=np.number)
    if numeric_df.empty: return {"error": "Нет числовых данных для анализа."}
    df_na_dropped = numeric_df.dropna()
    model = IsolationForest(contamination=0.05, random_state=42)
    predictions = model.fit_predict(df_na_dropped)
    outlier_indices = df_na_dropped.index[predictions == -1]
    outliers = df.loc[outlier_indices]
    return {"outlier_count": len(outliers), "outlier_preview": outliers.head(5).to_dict('records')}


@app.get("/download-cleaned/{file_id}")
async def download_cleaned_file(file_id: str):
    # ... (Ваш код эндпоинта /download-cleaned/ без изменений)
    if file_id not in file_metadata_storage: raise HTTPException(status_code=404, detail="File not found.")
    file_path = file_metadata_storage[file_id]['file_path']
    return FileResponse(path=file_path, media_type="text/csv", filename="cleaned_data.csv")