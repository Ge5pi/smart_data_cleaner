import io
import json
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Optional
import boto3
import numpy as np
import openai
import pandas as pd
import pinecone
import redis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sqlalchemy.orm import Session
from starlette import status
from starlette.responses import Response
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
import auth
import config
import crud
import database
import models
import schemas
from config import API_KEY, PINECONE_KEY

api_key = API_KEY
pinecone_key = PINECONE_KEY

try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        region_name=config.AWS_DEFAULT_REGION
    )

    redis_client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=5
    )
    redis_client.ping()
    print("--- Successfully connected to Redis ---")

except Exception as e:
    print(f"FATAL: Could not connect to AWS or Redis on startup. Error: {e}")

client = openai.OpenAI(api_key=api_key)
pc = pinecone.Pinecone(api_key=pinecone_key)
app = FastAPI(title="SODA API")

EMBEDDING_MODEL = "text-embedding-3-small"
AGENT_MODEL = "gpt-4o"
CRITIC_MODEL = "gpt-4o"
INDEX_NAME = "soda-index"
BATCH_SIZE = 100

user_router = APIRouter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=1536, metric='cosine')
index = pc.Index(INDEX_NAME)

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_df_from_s3(db: Session, file_id: str) -> pd.DataFrame:
    file_record = crud.get_file_by_uid(db, file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File record not found in DB.")

    try:
        obj = s3_client.get_object(Bucket=config.S3_BUCKET_NAME, Key=file_record.s3_path)
        file_content = io.BytesIO(obj['Body'].read())

        file_extension = Path(file_record.file_name).suffix.lower()
        if file_extension == '.csv':
            return pd.read_csv(file_content)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type in S3.")

    except Exception as e:
        print(f"Error reading file from S3: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read file from S3: {str(e)}")


@user_router.post("/users/register", response_model=schemas.User)
def register_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Пользователь с таким email уже зарегистрирован")
    return crud.create_user(db=db, user=user)


@user_router.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = auth.authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Неверный email или пароль",
                            headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@user_router.get("/files/me", response_model=list[schemas.File])
def read_user_files(db: Session = Depends(database.get_db),
                    current_user: models.User = Depends(auth.get_current_active_user)):
    return crud.get_files_by_user_id(db=db, user_id=current_user.id)


app.include_router(user_router, tags=["Users"])


@app.post("/sessions/start")
async def start_session(file_id: str = Form(...), current_user: models.User = Depends(auth.get_current_active_user),
                        db: Session = Depends(database.get_db)):
    print(f"User {current_user.email} (ID: {current_user.id}) is starting a new session.")

    # Проверяем, что файл существует в БД
    file_record = crud.get_file_by_uid(db, file_id)
    if not file_record or file_record.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="File not found or access denied.")

    session_id = str(uuid.uuid4())
    session_data = {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "file_id": file_id,
        "user_id": current_user.id
    }
    redis_client.setex(session_id, timedelta(hours=2), json.dumps(session_data))

    return {"session_id": session_id, "message": "Сессия успешно начата."}


def format_row(row_index: int, row: pd.Series) -> str:
    return f"Row-{row_index + 1}: " + " | ".join([f"{col}: {row[col]}" for col in row.index])


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(6))
def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [r.embedding for r in response.data]


def impute_missing_values_with_tabpfn(df: pd.DataFrame, target_column: str, max_samples: int = 1000) -> tuple:
    df_work = df.copy()
    original_series = df_work[target_column].copy()

    # 1. Проверка на наличие пропусков
    if not df_work[target_column].isna().any():
        return ("no_action", "В столбце нет пропущенных значений.", original_series)

    # 2. Разделение данных
    mask_missing = df_work[target_column].isna()
    df_complete = df_work[~mask_missing]
    df_missing = df_work[mask_missing]

    if len(df_complete) == 0:
        return ("skipped", "Нет полных данных для обучения модели.", original_series)

    # Ограничение выборки для производительности
    if len(df_complete) > max_samples:
        df_complete = df_complete.sample(n=max_samples, random_state=42)

    # 3. Выбор признаков для обучения
    feature_columns = [col for col in df_work.columns if col != target_column]
    useful_features = []
    for col in feature_columns:
        # Пропускаем столбцы с большим количеством пропусков или высокой кардинальностью
        if df_complete[col].isna().sum() / len(df_complete) > 0.5 or df_complete[col].nunique() > 100:
            continue
        useful_features.append(col)

    if not useful_features:
        return ("skipped", "Не найдено подходящих признаков для обучения.", original_series)

    # 4. Подготовка данных для модели
    X_train = df_complete[useful_features].copy()
    y_train = df_complete[target_column].copy()
    X_predict = df_missing[useful_features].copy()

    # Предобработка признаков
    for col in useful_features:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            # Заполняем пропуски в признаках как отдельную категорию
            X_train[col] = X_train[col].fillna('missing')
            X_predict[col] = X_predict[col].fillna('missing')

            all_values = pd.concat([X_train[col], X_predict[col]]).unique()
            le.fit(all_values)

            X_train[col] = le.transform(X_train[col])
            X_predict[col] = le.transform(X_predict[col])
        elif pd.api.types.is_numeric_dtype(X_train[col]):
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_predict[col] = X_predict[col].fillna(median_val)

    # 5. Обучение модели и предсказание
    try:
        # Решаем, использовать классификатор или регрессор
        is_classification = pd.api.types.is_object_dtype(y_train.dtype) or y_train.nunique() < 20

        if is_classification:
            model = TabPFNClassifier(device='cpu')
            le_target = LabelEncoder()
            y_train_encoded = le_target.fit_transform(y_train.astype(str))

            model.fit(X_train, y_train_encoded)
            predictions_encoded = model.predict(X_predict)
            predictions = le_target.inverse_transform(predictions_encoded)

            # Попытка вернуть исходный тип данных, если он был числовым
            if not pd.api.types.is_object_dtype(y_train.dtype):
                predictions = pd.Series(predictions).astype(y_train.dtype).values
        else:
            model = TabPFNRegressor(device='cpu')
            model.fit(X_train, y_train)
            predictions = model.predict(X_predict)
            # Приведение к исходному типу данных
            predictions = pd.Series(predictions).astype(y_train.dtype).values

        result = df_work[target_column].copy()
        result.loc[mask_missing] = predictions

        return ("success", "Пропуски успешно заполнены.", result)

    except Exception as e:
        # Логирование ошибки на бэкенде
        print(f"ERROR during TabPFN imputation for column '{target_column}': {e}")
        return ("error", f"Внутренняя ошибка модели: {str(e)}", original_series)


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


def execute_python_code(df: pd.DataFrame, code: str) -> tuple[pd.DataFrame, str]:
    """
    Выполняет Python-код над переданным DataFrame.
    Возвращает кортеж из (измененный DataFrame, строковый результат).
    """
    print(f"TOOL: Выполнение кода:\n---\n{code}\n---")
    local_scope = {"df": df.copy(), "pd": pd}
    try:
        # Попытка выполнить код и вернуть результат последнего выражения
        lines = code.strip().split('\n')
        exec_lines, eval_line = lines[:-1], lines[-1]
        if exec_lines:
            exec("\n".join(exec_lines), {}, local_scope)
        output = eval(eval_line, {}, local_scope)

        # Обновляем DataFrame из локальной области видимости
        final_df = local_scope['df']

        if isinstance(output, pd.DataFrame):
            return final_df, output.head(15).to_markdown()
        return final_df, str(output)

    except Exception:
        # Если eval не сработал, пробуем просто выполнить код (например, для присваивания)
        try:
            exec(code, {}, local_scope)
            final_df = local_scope['df']
            return final_df, "Код выполнен, DataFrame обновлен."
        except Exception as exec_e:
            # Если и это не сработало, возвращаем ошибку и исходный DataFrame
            return df, f"Ошибка выполнения кода: {exec_e}"


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


def save_dataframe_to_file(db: Session, file_id: str, df_to_save: pd.DataFrame) -> str:
    print(f"TOOL: Сохранение файла {file_id}...")

    file_record = crud.get_file_by_uid(db, file_id)
    if not file_record:
        return "Ошибка: Запись о файле не найдена в БД."

    try:
        with io.StringIO() as csv_buffer:
            df_to_save.to_csv(csv_buffer, index=False)
            # Загружаем содержимое буфера в S3
            s3_client.put_object(Body=csv_buffer.getvalue(), Bucket=config.S3_BUCKET_NAME, Key=file_record.s3_path)
        return f"Файл '{file_record.file_name}' успешно сохранен."
    except Exception as e:
        return f"Ошибка при сохранении файла в S3: {e}"


def answer_question_from_context(file_id: str, query: str) -> str:
    print(f"TOOL: RAG-запрос для файла {file_id}: '{query}'")
    # run_rag_pipeline уже использует file_id для фильтрации в Pinecone
    return run_rag_pipeline(file_id, query)


tools_definition = [
    {"type": "function", "function": {
        "name": "execute_python_code",
        "description": "Выполняет Python-код для манипуляций с DataFrame `df`. Используется для расчетов, агрегации, "
                       "фильтрации, модификации данных.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Код на Python для выполнения над DataFrame 'df'."}
            },
            "required": ["code"]
        }
    }},
    {"type": "function", "function": {
        "name": "answer_question_from_context",
        "description": "Ищет в файле строки по смысловому содержанию и отвечает на вопрос на их основе. Пример: 'Что "
                       "можешь рассказать о заказе A-123?', 'Опиши клиента с самой большой суммой покупки'.",
        "parameters": {
            "type": "object",
            "properties": {
                # Модель должна передавать только поисковый запрос
                "query": {"type": "string", "description": "Вопрос для поиска по смыслу в документе."}
            },
            "required": ["query"]
        }
    }},
    {"type": "function", "function": {
        "name": "save_dataframe_to_file",
        "description": "Сохраняет текущий DataFrame `df` в исходный файл. Использовать только по прямой и явной "
                       "просьбе пользователя, например: 'Сохрани результат'.",
        "parameters": {
            "type": "object",
            "properties": {}  # Аргументы не требуются от модели
        }
    }},
]
available_functions = {
    "execute_python_code": execute_python_code,
    "answer_question_from_context": answer_question_from_context,
    "save_dataframe_to_file": save_dataframe_to_file,
}


@app.post("/upload/", tags=["File Operations"])
async def upload_file(
        file: UploadFile = File(...),
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    file_id = str(uuid.uuid4())
    original_filename = file.filename
    file_extension = Path(original_filename).suffix.lower()

    if file_extension not in ['.csv', '.xlsx', '.xls']:
        raise HTTPException(status_code=400, detail="Неподдерживаемый тип файла.")

    s3_path = f"uploads/{current_user.id}/{file_id}/{original_filename}"

    try:
        contents = await file.read()
        s3_client.put_object(Body=contents, Bucket=config.S3_BUCKET_NAME, Key=s3_path)

        db_file = crud.create_user_file(db=db, user_id=current_user.id, file_uid=file_id, file_name=original_filename,
                                        s3_path=s3_path)

        file_stream = io.BytesIO(contents)
        if file_extension == '.csv':
            df = pd.read_csv(file_stream, encoding="utf-8")
        else:
            df = pd.read_excel(file_stream)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке или чтении файла: {e}")

    df = df.replace([np.inf, -np.inf], np.nan)

    # Логика индексации в Pinecone
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

    analysis = [{"column": col, "dtype": str(df[col].dtype), "nulls": int(df[col].isna().sum()),
                 "unique": int(df[col].nunique())} for col in df.columns]
    preview_data = df.head(10).fillna("null").to_dict(orient="records")
    return {"columns": analysis, "preview": preview_data, "file_id": file_id, "file_name": original_filename}


@app.post("/sessions/ask", tags=["AI Agent"])
async def session_ask(session_id: str = Form(...), query: str = Form(...), db: Session = Depends(database.get_db)):
    """
    Обрабатывает запрос пользователя в рамках сессии, вызывает инструменты,
    оценивает ответ с помощью критика и при необходимости улучшает его.
    """
    session_data_json = redis_client.get(session_id)
    if not session_data_json:
        raise HTTPException(status_code=404, detail="Сессия не найдена или истекла.")

    session_data = json.loads(session_data_json)
    file_id = session_data["file_id"]
    messages = session_data["messages"]

    # 2. Загрузка DataFrame из S3
    df = get_df_from_s3(db, file_id)

    # 3. Подготовка контекста для модели
    buf = io.StringIO()
    df.info(buf=buf)
    df_info = buf.getvalue()
    df_head = df.head().to_markdown()
    contextual_query = (
        f"Контекст данных:\n1. Схема данных (df.info()):\n```\n{df_info}```\n"
        f"2. Первые 5 строк (df.head()):\n```\n{df_head}```\n---\n"
        f"Вопрос пользователя: {query}"
    )

    messages.append({"role": "user", "content": contextual_query})

    try:
        final_answer = ""

        # 4. Основной цикл вызова AI-агента и инструментов
        for _ in range(5):
            response = client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                tools=tools_definition,
                tool_choice="auto"
            )
            response_message = response.choices[0].message

            # --- ИСПРАВЛЕНИЕ: Преобразуем объект в словарь ---
            # Вместо messages.append(response_message)
            messages.append(response_message.model_dump())

            if not response_message.tool_calls:
                final_answer = response_message.content
                break

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "execute_python_code":
                    df, function_response = execute_python_code(df=df, code=function_args.get("code"))
                elif function_name == "save_dataframe_to_file":
                    function_response = save_dataframe_to_file(db=db, file_id=file_id, df_to_save=df)
                elif function_name == "answer_question_from_context":
                    function_response = answer_question_from_context(file_id=file_id, query=function_args.get("query"))
                else:
                    function_response = f"Ошибка: неизвестный инструмент {function_name}"

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                })

        if not final_answer:
            last_response = client.chat.completions.create(model=AGENT_MODEL, messages=messages)
            final_answer = last_response.choices[0].message.content
            messages.append({"role": "assistant", "content": final_answer})

        # 5. Вызов критика для оценки ответа
        evaluation = get_critic_evaluation(query, final_answer)
        print(f"Critic Evaluation: {evaluation}")

        # 6. Улучшение ответа
        if evaluation.get("accuracy", 5) < 4:
            print("Ответ неудовлетворительный. Запускается модель-улучшатель...")
            # Передаем в историю только словари, а не объекты
            history_for_refiner = [msg for msg in messages if isinstance(msg, dict)]
            refined_answer = get_refined_answer(
                history=history_for_refiner,
                original_answer=final_answer,
                feedback=evaluation.get("feedback"),
                suggestion=evaluation.get("suggestion", "")
            )
            final_answer = refined_answer
            messages.append({"role": "assistant", "content": final_answer})

        # 7. Обновление сессии в Redis и возврат ответа
        session_data["messages"] = messages
        redis_client.setex(session_id, timedelta(hours=2), json.dumps(session_data))

        return {"answer": final_answer, "evaluation": evaluation}

    except Exception as e:
        session_data["messages"] = messages
        redis_client.setex(session_id, timedelta(hours=2), json.dumps(session_data))
        print(f"Error in session_ask (session: {session_id}): {e}")
        raise HTTPException(status_code=500, detail=f"Произошла внутренняя ошибка: {str(e)}")


@app.post("/analyze/", tags=["File Operations"])
async def analyze_csv(file_id: str = Form(...), db: Session = Depends(database.get_db)):
    # Используем новую функцию-помощник
    df = get_df_from_s3(db, file_id)

    analysis = [{"column": col, "dtype": str(df[col].dtype), "nulls": int(df[col].isna().sum()),
                 "unique": int(df[col].nunique()), "sample_values": df[col].dropna().astype(str).unique()[:3].tolist()}
                for col in df.columns]
    return {"columns": analysis}


@app.post("/impute-missing/", tags=["Data Cleaning"])
async def impute_missing_values_endpoint(file_id: str = Form(...), columns: Optional[str] = Form(None),
                                         db: Session = Depends(database.get_db)):
    """Заполняет пропуски в данных и сохраняет измененный файл обратно в S3."""
    df = get_df_from_s3(db, file_id)

    if not columns:
        raise HTTPException(status_code=400, detail="No columns selected for imputation.")

    selected_columns = json.loads(columns)
    if not isinstance(selected_columns, list) or not selected_columns:
        raise HTTPException(status_code=400, detail="Invalid columns format.")

    df_imputed = df.copy()
    final_processing_results, final_missing_before, final_missing_after = {}, {}, {}

    for col in selected_columns:
        if col not in df.columns:
            final_processing_results[col], final_missing_before[col], final_missing_after[
                col] = "Ошибка: Столбец не найден", "N/A", "N/A"
            continue

        missing_before = int(df[col].isna().sum())
        _status, message, new_series = impute_missing_values_with_tabpfn(df,
                                                                         col)  # `impute_missing_values_with_tabpfn` находится в секции AI/ML
        df_imputed[col] = new_series

        final_missing_before[col] = missing_before
        final_missing_after[col] = int(new_series.isna().sum())
        final_processing_results[col] = message

    try:
        file_record = crud.get_file_by_uid(db, file_id)
        with io.StringIO() as csv_buffer:
            df_imputed.to_csv(csv_buffer, index=False)
            s3_client.put_object(Body=csv_buffer.getvalue(), Bucket=config.S3_BUCKET_NAME, Key=file_record.s3_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save imputed file to S3: {str(e)}")

    return {
        "processing_results": final_processing_results, "missing_before": final_missing_before,
        "missing_after": final_missing_after, "preview": df_imputed.head(10).fillna("null").to_dict(orient="records"),
    }


@app.post("/outliers/", tags=["Data Cleaning"])
async def detect_outliers_endpoint(file_id: str = Form(...), columns: Optional[str] = Form(None),
                                   db: Session = Depends(database.get_db)):
    """Находит выбросы в числовых столбцах."""
    df = get_df_from_s3(db, file_id)

    if columns:
        selected_columns = json.loads(columns)
    else:
        selected_columns = df.select_dtypes(include=np.number).columns.tolist()

    if not selected_columns:
        return {"outlier_count": 0, "outlier_preview": []}

    valid_columns = [col for col in selected_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not valid_columns:
        return {"outlier_count": 0, "outlier_preview": []}

    numeric_df = df[valid_columns].dropna()
    if numeric_df.empty:
        return {"outlier_count": 0, "outlier_preview": []}

    model = IsolationForest(contamination=0.05, random_state=42)
    predictions = model.fit_predict(numeric_df)
    outlier_indices = numeric_df.index[predictions == -1]
    outliers_df = df.loc[outlier_indices]
    outliers_preview = outliers_df.head(5).fillna("null").to_dict('records')
    return {"outlier_count": len(outliers_df), "outlier_preview": outliers_preview}


@app.get("/download-cleaned/{file_id}", tags=["File Operations"])
async def download_cleaned_file(file_id: str, db: Session = Depends(database.get_db)):
    """Отдает пользователю последнюю версию файла напрямую из S3."""
    file_record = crud.get_file_by_uid(db, file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File record not found in DB.")

    try:
        obj = s3_client.get_object(Bucket=config.S3_BUCKET_NAME, Key=file_record.s3_path)
        return Response(
            content=obj['Body'].read(),
            media_type='text/csv',
            headers={"Content-Disposition": f'attachment; filename="{file_record.file_name}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch file from S3: {e}")


@app.post("/analyze-existing/", tags=["File Operations"])
async def analyze_existing_file(file_id: str = Form(...), db: Session = Depends(database.get_db)):
    df = get_df_from_s3(db, file_id)
    analysis = [{"column": col, "dtype": str(df[col].dtype), "nulls": int(df[col].isna().sum()),
                 "unique": int(df[col].nunique())} for col in df.columns]
    preview_data = df.head(10).fillna("null").to_dict(orient="records")
    return {"columns": analysis, "preview": preview_data}


@app.post("/chart-data/")
async def get_chart_data(
        file_id: str = Form(...),
        chart_type: str = Form(...),
        column1: str = Form(...),
        column2: Optional[str] = Form(None),
        column3: Optional[str] = Form(None),
        db: Session = Depends(database.get_db)
):
    df = get_df_from_s3(db, file_id)

    chart_data = {}

    try:
        if chart_type in ["histogram", "pie"]:
            # ... (логика без изменений)
            if df[column1].dtype == 'object':
                counts = df[column1].dropna().value_counts().nlargest(15)
            else:
                counts = df[column1].dropna().value_counts()
            chart_data = {"labels": counts.index.astype(str).tolist(), "values": counts.values.tolist()}

        elif chart_type in ["line", "area"] and column2:
            try:
                df[column1] = pd.to_datetime(df[column1])
                line_df = df[[column1, column2]].dropna().sort_values(by=column1)
                chart_data = {
                    "labels": line_df[column1].dt.strftime('%Y-%m-%d').tolist(),
                    "values": line_df[column2].tolist(),
                }
            except Exception:
                raise HTTPException(status_code=400, detail=f"Не удалось преобразовать столбец '{column1}' в дату.")

        elif chart_type == "scatter" and column2:
            scatter_df = df[[column1, column2]].dropna()
            chart_data = {"points": scatter_df.to_dict(orient='records')}

        elif chart_type == "bubble" and column2 and column3:
            bubble_df = df[[column1, column2, column3]].dropna()
            for col in [column1, column2, column3]:
                if not pd.api.types.is_numeric_dtype(bubble_df[col]):
                    raise HTTPException(status_code=400,
                                        detail=f"Для пузырьковой диаграммы все столбцы должны быть числовыми. Столбец '{col}' не является числовым.")

            chart_data = {"points": bubble_df.to_dict(orient='records')}

        elif chart_type == "boxplot":
            if not pd.api.types.is_numeric_dtype(df[column1]):
                raise HTTPException(status_code=400,
                                    detail=f"Для 'Ящика с усами' столбец '{column1}' должен быть числовым.")

            stats = df[column1].describe()
            q1 = stats['25%']
            q3 = stats['75%']
            iqr = q3 - q1
            lower_whisker = q1 - 1.5 * iqr
            upper_whisker = q3 + 1.5 * iqr

            chart_data = {
                "min": stats['min'], "q1": q1, "median": stats['50%'],
                "q3": q3, "max": stats['max'],
                "outliers": df[(df[column1] < lower_whisker) | (df[column1] > upper_whisker)][column1].tolist()
            }

        else:
            raise HTTPException(status_code=400, detail="Неверный тип графика или нехватка данных.")

        return {"chart_type": chart_type, "data": chart_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при подготовке данных для графика: {str(e)}")
