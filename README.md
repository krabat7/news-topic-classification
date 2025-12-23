# Классификация новостей по темам (AG News)

Модель классифицирует новость (title + description) в одну из 4 тем: World, Sports, Business, Sci/Tech.
Датасет: AG News (HuggingFace Datasets). Язык: English.

## Формат входа/выхода

Input (JSON):

1. {"title": "string", "description": "string"}
2. {"text": "string"}

Если text отсутствует:
text = title + " [SEP] " + description

Output (JSON):
{
"label": "World|Sports|Business|Sci/Tech",
"probs": {"World": 0.0, "Sports": 0.0, "Business": 0.0, "Sci/Tech": 0.0}
}

## Метрики

Accuracy, Macro-F1, Micro-F1

## Валидация

Стандартный split AG News train/test.
Train делится на train/val = 85/15 (stratified), фиксируется seed и конфиги.

## Setup

Python 3.10-3.12. Менеджер зависимостей poetry.

1. Установить poetry: https://python-poetry.org/docs/#installation
2. В корне репозитория:
   poetry env use python3.12 (или нужную версию в диапазоне 3.10-3.12)
   poetry install
   poetry run pre-commit install
   poetry run pre-commit run -a
3. Настроить DVC remote (пример для локального каталога вне репозитория):
   dvc remote add -d storage ../dvcstore
   dvc pull data/ag_news.dvc
   Если remote пустой, код скачает датасет из HuggingFace при первом запуске и сохранит его в data/ag_news.

## Data (DVC)

Трекер данных: data/ag_news.dvc. Кэш и сами данные в git не хранятся.
ensure_data() в коде сперва вызывает dvc pull, затем при отсутствии данных скачает ag_news из HuggingFace и сохранит в data/ag_news.

## Logging (MLflow)

Адрес по умолчанию: mlruns (локальный каталог)
Поднять UI локально:
poetry run mlflow ui --backend-store-uri file:mlruns --host 127.0.0.1 --port 8080
Переопределение через Hydra:
poetry run python -m news_topic_classifier.commands train logging.tracking_uri=http://127.0.0.1:8080

## Train

Базовый запуск:
poetry run python -m news_topic_classifier.commands train
Пример с оверрайдами:
poetry run python -m news_topic_classifier.commands train trainer.max_epochs=1 batch_size=32
Лучший чекпоинт сохраняется в outputs/checkpoints/best.ckpt
Словарь сохраняется в outputs/artifacts/vocab.json

## Infer

После обучения:
poetry run python -m news_topic_classifier.commands infer infer.ckpt_path=outputs/checkpoints/best.ckpt
Пути и тексты задаются в configs/infer.yaml. В инференсе используется сохраненный vocab из outputs/artifacts/vocab.json.

## Примечания

- Перед train/infer нужно понять, что DVC remote доступен (или тогда нужноо позвоилить коду скачать HF датасет локально).
- MLflow по умолчанию пишет в mlruns и для внешнего трекинга указать logging.tracking_uri.
