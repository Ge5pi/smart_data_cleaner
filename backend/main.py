import uuid
from contextlib import asynccontextmanager
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
from starlette import status
from starlette.responses import FileResponse
import openai
import pinecone
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier, TabPFNRegressor
from config import API_KEY, PINECONE_KEY, DATABASE_URL
import crud, models, schemas, database
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends
from sqlalchemy.orm import Session
from fastapi import APIRouter
from datetime import timedelta
import auth


api_key = API_KEY
pinecone_key = PINECONE_KEY


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Populating file metadata cache on startup ---")
    with database.SessionLocal() as db:
        try:
            all_files_from_db = crud.get_all_files(db)
            for file_record in all_files_from_db:
                file_id = file_record.file_uid
                output_path = TEMP_DIR / f"{file_id}.csv"
                if output_path.exists():
                    file_metadata_storage[file_id] = {
                        "file_path": str(output_path),
                        "file_name": file_record.file_name,
                    }
                    print(f"  - Cached metadata for file: {file_id}")
                else:
                    print(f"  - Skipped caching for missing file on disk: {file_id}")
        except Exception as e:
            print(f"Error during cache population: {e}")

    print("--- Cache population complete. Application is ready. ---")
    yield
    file_metadata_storage.clear()
    print("--- Cleared file metadata cache on shutdown ---")

client = openai.OpenAI(api_key=api_key)
pc = pinecone.Pinecone(api_key=pinecone_key)
app = FastAPI()

EMBEDDING_MODEL = "text-embedding-3-small"
AGENT_MODEL = "gpt-4o"
CRITIC_MODEL = "gpt-4o"
INDEX_NAME = "soda-index"
BATCH_SIZE = 100

user_router = APIRouter(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

TEMP_DIR = Path("temp_cleaned_files")
TEMP_DIR.mkdir(exist_ok=True)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=1536, metric='cosine')
index = pc.Index(INDEX_NAME)

file_metadata_storage = {}
session_cache = {}


@user_router.post("/users/register", response_model=schemas.User)
def register_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)

    if db_user:
        raise HTTPException(status_code=400, detail="Пользователь с таким email уже зарегистрирован")
    new_user = crud.create_user(db=db, user=user)
    return new_user


@user_router.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = auth.authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный email или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@user_router.get("/files/me", response_model=list[schemas.File])
def read_user_files(
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    return crud.get_files_by_user_id(db=db, user_id=current_user.id)


@app.post("/sessions/start")
async def start_session(file_id: str = Form(...), current_user: models.User = Depends(auth.get_current_active_user)):
    print(f"Пользователь {current_user.email} (ID: {current_user.id}) начинает новую сессию.")

    if file_id not in file_metadata_storage:
        raise HTTPException(status_code=404, detail="Файл с таким ID не найден.")
    session_id = str(uuid.uuid4())
    df = pd.read_csv(file_metadata_storage[file_id]['file_path'])

    session_cache[session_id] = {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "dataframe": df,
        "file_id": file_id,
        "user_id": current_user.id
    }
    return {"session_id": session_id, "message": "Сессия успешно начата."}


app.include_router(user_router)

def format_row(row_index: int, row: pd.Series) -> str:
    return f"Row-{row_index + 1}: " + " | ".join([f"{col}: {row[col]}" for col in row.index])


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(6))
def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [r.embedding for r in response.data]


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
    useful_features, label_encoders = [], {}
    for col in feature_columns:
        if df_complete[col].isna().sum() / len(df_complete) > 0.5 or df_complete[col].nunique() > 100:
            continue
        useful_features.append(col)
    if not useful_features:
        return df_work[target_column]
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
    except Exception:
        return df_work[target_column]


def get_critic_evaluation(query: str, answer: str) -> dict:
    critic_prompt = f""" You are a meticulous AI data analyst critic. Your task is to evaluate a generated answer 
    based on a user's query. Provide your evaluation in a structured JSON format. 

        **User Query:**
        {query}

        **Answer to Evaluate:**
        {answer}

        **Evaluation Criteria:** 1.  **relevance**: Is the answer directly related to the user's query? (1-5, 
        5 is most relevant) 2.  **completeness**: Does the answer fully address all parts of the query? (1-5, 
        5 is most complete) 3.  **accuracy**: How likely is the answer to be factually correct in a pandas DataFrame 
        context? Does it seem plausible? (1-5, 5 is most accurate) 4.  **feedback**: Provide CONCISE and ACTIONABLE 
        feedback. - If the answer is good, write "The answer is sufficient.". - If the answer is bad, explain WHY it 
        is bad. For example: "The answer hallucinates information not present in the query" or "The calculation seems 
        incorrect for the requested metric." 5.  **suggestion**: If the answer is poor, suggest a better approach. 
        For example: "It would be better to use `df.groupby('category')['sales'].sum()`" or "A better approach would 
        be to use the RAG tool to find the specific row." 

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


def get_refined_answer(history: list, original_answer: str, feedback: str, suggestion: str) -> str:
    refiner_prompt = f""" You are an expert data analyst. Your previous attempt to answer a user's question was 
    flawed. A critic has provided feedback. Your task is to generate a new, final, and correct answer that 
    incorporates this feedback. 

        **Original Conversation History:**
        {json.dumps(history, indent=2, ensure_ascii=False)}

        **Your previous (unsatisfactory) answer:**
        {original_answer}

        **Critic's Feedback (What was wrong):**
        {feedback}

        **Critic's Suggestion (How to fix it):**
        {suggestion}

        **Your Task:** Generate a new, improved, and complete answer. The new answer must directly address the user's 
        last question and fix the issues raised by the critic. Do not repeat the mistakes. Present the final result 
        clearly to the user. """
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[{"role": "user", "content": refiner_prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """Ты — элитный AI-аналитик данных SODA (Strategic Operations & Data Analyst). Твоя работа — помогать 
пользователю анализировать данные в pandas DataFrame `df`. 

**Твой мыслительный процесс должен быть следующим:** 1.  **Анализ Запроса:** Внимательно изучи вопрос пользователя и 
предоставленный контекст (`df.info()`, `df.head()`). Определи ключевую цель пользователя: он хочет изменить данные, 
получить агрегированную информацию, найти что-то конкретное или понять смысл данных? 2.  **Выбор Инструмента:** 
Основываясь на цели, выбери ОДИН из доступных инструментов. *   Используй `execute_python_code`, если запрос требует: 
- Математических операций (среднее, сумма, группировка, подсчет). - Фильтрации или сортировки данных (`df[...]`, 
`.query()`, `.sort_values()`). - Модификации DataFrame (создание новых столбцов, удаление, заполнение пропусков). - 
Получения информации о структуре данных (`df.describe()`, `df.corr()`). *   Используй `answer_question_from_context`, 
если запрос требует: - Найти и описать конкретные строки по смысловому содержанию (например, "Что известно о клиенте 
Иване Петрове?", "Найди записи, связанные с 'неудачной доставкой'"). - Ответить на общие вопросы, ответ на которые 
может содержаться в одной или нескольких строках как текст. *   Используй `save_dataframe_to_file` **ТОЛЬКО** если 
пользователь явно и недвусмысленно попросил "сохрани файл", "запиши изменения" или аналогичное. Всегда уточняй перед 
сохранением, если не уверен. 3.  **Формулировка Ответа:** После выполнения инструмента, предоставь пользователю 
четкий и лаконичный ответ. *   Если ты использовал `execute_python_code`, покажи результат (обычно это 
markdown-таблица) и кратко объясни, что он означает. *   Если ты получил ошибку при выполнении кода, проанализируй 
ее, исправь свой код и попробуй снова. Не повторяй одну и ту же ошибку. *   Отвечай всегда на языке запроса. """


def execute_python_code(session_id: str, code: str) -> str:
    print(f"TOOL (session: {session_id}): Выполнение кода:\n---\n{code}\n---")
    if session_id not in session_cache:
        return "Ошибка: Сессия не найдена."
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
    query_embedding = get_embeddings([query])[0]
    search_results = index.query(vector=query_embedding, top_k=7, filter={"file_id": file_id}, include_metadata=True)
    context = " "
    if search_results.get('matches'):
        for match in search_results['matches']:
            context += match['metadata']['original_text'] + "\n---\n"
    if not context:
        return "Не удалось найти релевантную информацию в файле."
    rag_messages = [{"role": "system", "content": "Ответь на вопрос пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на "
                                                  "предоставленном контексте."}, {"role": "user", "content":
        f"Контекст:\n{context}\n\nВопрос: {query}"}]
    response = client.chat.completions.create(model=AGENT_MODEL, messages=rag_messages, temperature=0.0)
    return response.choices[0].message.content


def save_dataframe_to_file(session_id: str) -> str:
    print(f"TOOL (session: {session_id}): Сохранение файла...")
    if session_id not in session_cache:
        return "Ошибка: Сессия не найдена."

    file_id = session_cache[session_id].get("file_id")
    if not file_id:
        return "Ошибка: file_id не найден в сессии."

    file_path_str = file_metadata_storage.get(file_id, {}).get("file_path")
    if not file_path_str:
        return "Ошибка: Путь к файлу не найден в метаданных."

    # ПРОВЕРКА БЕЗОПАСНОСТИ: Убедимся, что путь находится внутри нашей временной директории
    file_path = Path(file_path_str).resolve()
    if TEMP_DIR.resolve() not in file_path.parents:
        return "Ошибка безопасности: Попытка записи файла за пределами разрешенной директории."

    try:
        df_to_save = session_cache[session_id]["dataframe"]
        df_to_save.to_csv(file_path, index=False)
        return f"Файл успешно сохранен по пути: {file_path_str}. Все изменения применены."
    except Exception as e:
        return f"Ошибка при сохранении файла: {e}"


def answer_question_from_context(session_id: str, query: str) -> str:
    print(f"TOOL (session: {session_id}): RAG-запрос: '{query}'")
    file_id = session_cache.get(session_id, {}).get("file_id")
    if not file_id:
        return "Ошибка: не найден исходный file_id."
    return run_rag_pipeline(file_id, query)


tools_definition = [
    {"type": "function", "function": {
        "name": "execute_python_code",
        "description": "Выполняет Python-код для манипуляций с DataFrame `df`. Используется для расчетов, агрегации, "
                       "фильтрации, модификации данных.",
        "parameters": {
            "type": "object", "properties": {"session_id": {"type": "string"}, "code": {"type": "string"}},
            "required": ["session_id", "code"]
        }
    }},
    {"type": "function", "function": {
        "name": "answer_question_from_context",
        "description": "Ищет в файле строки по смысловому содержанию и отвечает на вопрос на их основе. Пример: 'Что "
                       "можешь рассказать о заказе A-123?', 'Опиши клиента с самой большой суммой покупки'. НЕ "
                       "использовать для математики или агрегации.",
        "parameters": {
            "type": "object", "properties": {"session_id": {"type": "string"}, "query": {"type": "string"}},
            "required": ["session_id", "query"]
        }
    }},
    {"type": "function", "function": {
        "name": "save_dataframe_to_file",
        "description": "Сохраняет текущий DataFrame `df` в исходный файл. Использовать только по прямой и явной "
                       "просьбе пользователя, например: 'Сохрани результат'.",
        "parameters": {
            "type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]
        }
    }},
]
available_functions = {
    "execute_python_code": execute_python_code,
    "answer_question_from_context": answer_question_from_context,
    "save_dataframe_to_file": save_dataframe_to_file,  # Регистрируем новый инструмент
}

@app.post("/upload/")
async def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    file_id = str(uuid.uuid4())
    output_path = TEMP_DIR / f"{file_id}.csv"
    original_filename = file.filename

    crud.create_user_file(
        db=db,
        user_id=current_user.id,
        file_uid=file_id,
        file_name=original_filename
    )

    with open(output_path, "wb") as buffer:
        buffer.write(await file.read())

    df = pd.read_csv(output_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv(output_path, index=False)

    file_metadata_storage[file_id] = {
        "file_path": str(output_path),
        "file_name": original_filename,
        "total_rows": len(df),
        "column_names": df.columns.tolist(),
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
            for j, vector in enumerate(batch):
                vector["values"] = embeddings[j]
            index.upsert(vectors=batch)
        except Exception as e:
            print(f"Ошибка при обработке пачки {i}-{i + BATCH_SIZE}: {e}")

    preview_data = df.head(10).fillna("null").to_dict(orient="records")
    return {"preview": preview_data, "file_id": file_id}


@app.post("/sessions/start")
async def start_session(file_id: str = Form(...)):
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
    if session_id not in session_cache:
        raise HTTPException(status_code=404, detail="Сессия не найдена.")
    df = session_cache[session_id]["dataframe"]
    buf = io.StringIO()
    df.info(buf=buf)
    df_info = buf.getvalue()
    df_head = df.head().to_markdown()
    contextual_query = f"Контекст данных:\n1. Схема данных (df.info()):\n```\n{df_info}```\n2. Первые 5 строк (" \
                       f"df.head()):\n```\n{df_head}```\n---\nВопрос пользователя: {query} "
    messages = session_cache[session_id]["messages"]
    messages.append({"role": "user", "content": contextual_query})
    try:
        initial_answer = ""
        for _ in range(5):
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
        evaluation = get_critic_evaluation(query, initial_answer)
        print(f"Critic Evaluation: {evaluation}")
        final_answer = initial_answer
        if evaluation.get("accuracy", 5) < 4:
            print("Ответ неудовлетворительный. Запускается модель-улучшатель...")
            final_answer = get_refined_answer(
                messages,
                initial_answer,
                evaluation.get("feedback"),
                evaluation.get("suggestion", "")
            )
            messages.append({"role": "assistant", "content": final_answer})
        session_cache[session_id]["messages"] = messages
        return {"answer": final_answer, "evaluation": evaluation}
    except Exception as e:
        session_cache[session_id]["messages"] = messages
        raise HTTPException(status_code=500, detail=f"Произошла внутренняя ошибка: {str(e)}")


@app.post("/analyze/")
async def analyze_csv(file_id: str = Form(...)):
    if file_id not in file_metadata_storage:
        raise HTTPException(status_code=404, detail="File not found.")
    df = pd.read_csv(file_metadata_storage[file_id]['file_path'])
    analysis = [{"column": col, "dtype": str(df[col].dtype), "nulls": int(df[col].isna().sum()),
                 "unique": int(df[col].nunique()), "sample_values": df[col].dropna().astype(str).unique()[:3].tolist()}
                for col in df.columns]
    return {"columns": analysis}


@app.post("/impute-missing/")
async def impute_missing_values(file_id: str = Form(...), columns: Optional[str] = Form(None)):
    if file_id not in file_metadata_storage:
        raise HTTPException(status_code=404, detail="File not found.")
    df = pd.read_csv(file_metadata_storage[file_id]['file_path'])
    selected_columns = json.loads(columns) if columns else [col for col in df.columns if df[col].isna().any()]
    if not selected_columns:
        return {"error": "Нет столбцов с пропущенными значениями."}
    df_imputed = df.copy()
    for col in selected_columns:
        if df[col].isna().any():
            df_imputed[col] = impute_missing_values_with_tabpfn(df, col)
    df_imputed.to_csv(file_metadata_storage[file_id]['file_path'], index=False)  # Сохраняем изменения
    return {"message": "Заполнение пропусков завершено."}


@app.post("/outliers/")
async def detect_outliers(file_id: str = Form(...), columns: Optional[str] = Form(None)):
    if file_id not in file_metadata_storage:
        raise HTTPException(status_code=404, detail="File not found.")
    df = pd.read_csv(file_metadata_storage[file_id]['file_path'])
    selected_columns = json.loads(columns) if columns else df.select_dtypes(include=np.number).columns.tolist()
    numeric_df = df[selected_columns].select_dtypes(include=np.number)
    if numeric_df.empty:
        return {"error": "Нет числовых данных для анализа."}
    df_na_dropped = numeric_df.dropna()
    model = IsolationForest(contamination=0.05, random_state=42)
    predictions = model.fit_predict(df_na_dropped)
    outlier_indices = df_na_dropped.index[predictions == -1]
    outliers = df.loc[outlier_indices]
    return {"outlier_count": len(outliers), "outlier_preview": outliers.head(5).to_dict('records')}


@app.get("/download-cleaned/{file_id}")
async def download_cleaned_file(file_id: str):
    if file_id not in file_metadata_storage:
        raise HTTPException(status_code=404, detail="File not found.")
    file_path = file_metadata_storage[file_id]['file_path']
    return FileResponse(path=file_path, media_type="text/csv", filename="cleaned_data.csv")


@app.post("/analyze-existing/")
async def analyze_existing_file(file_id: str = Form(...)):
    file_path = f"{TEMP_DIR}//{file_id}.csv"

    try:
        df = pd.read_csv(file_path)
        analysis = [{"column": col, "dtype": str(df[col].dtype), "nulls": int(df[col].isna().sum()),
                     "unique": int(df[col].nunique()),
                     "sample_values": df[col].dropna().astype(str).unique()[:3].tolist()}
                    for col in df.columns]
        preview_data = df.head(10).fillna("null").to_dict(orient="records")

        return {"columns": analysis, "preview": preview_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or analyze file: {str(e)}")


@app.post("/chart-data/")
async def get_chart_data(
        file_id: str = Form(...),
        chart_type: str = Form(...),
        column1: str = Form(...),
        column2: Optional[str] = Form(None),
):
    """Готовит данные для указанного типа графика."""
    if file_id not in file_metadata_storage:
        raise HTTPException(status_code=404, detail="Файл не найден в кэше сервера.")

    file_path = file_metadata_storage[file_id]['file_path']
    df = pd.read_csv(file_path)

    chart_data = {}

    try:

        if chart_type == "histogram" or chart_type == "pie":
            if df[column1].dtype == 'object':
                counts = df[column1].dropna().value_counts().nlargest(15)
            else:  # для чисел
                counts = df[column1].dropna().value_counts()

            chart_data = {
                "labels": counts.index.astype(str).tolist(),
                "values": counts.values.tolist(),
            }

        elif chart_type == "scatter" and column2:
            # Для диаграммы рассеяния
            scatter_df = df[[column1, column2]].dropna()
            chart_data = {
                "points": scatter_df.to_dict(orient='records')
            }

        elif chart_type == "line" and column2:
            try:
                df[column1] = pd.to_datetime(df[column1])
                line_df = df[[column1, column2]].dropna().sort_values(by=column1)

                chart_data = {
                    "labels": line_df[column1].dt.strftime('%Y-%m-%d').tolist(),  # Форматируем даты
                    "values": line_df[column2].tolist(),
                }
            except Exception:
                raise HTTPException(status_code=400,
                                    detail=f"Не удалось преобразовать столбец '{column1}' в дату. Выберите другой "
                                           f"столбец для оси X.")

        else:
            raise HTTPException(status_code=400, detail="Неверный тип графика или нехватка данных.")

        return {"chart_type": chart_type, "data": chart_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при подготовке данных для графика: {str(e)}")
