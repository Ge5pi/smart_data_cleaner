from celery import Celery
import certifi
import ssl
REDIS_URL = "rediss://default:Ac4_AAIncDE1YmEyZjIyNGM5Zjc0ZjA5OWY3NjRlZmEwZmU4ZTY0NHAxNTI3OTk@proud-coral-52799.upstash.io:6379"

celery_app = Celery("worker")

celery_app.conf.update(
    broker_url=REDIS_URL,
    broker_use_ssl={"ssl_cert_reqs": ssl.CERT_REQUIRED},
    result_backend=REDIS_URL,
    redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_REQUIRED},
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True,
)

celery_app.conf.task_routes = {
    "tasks.*": {"queue": "default"},
}
