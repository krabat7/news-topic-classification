# Классификация новостей по темам (AG News)

Модель классифицирует новость (title + description) в одну из 4 тем: World, Sports, Business, Sci/Tech.
Датасет: AG News (HuggingFace Datasets). Язык: English.

## Формат входа/выхода

Input (JSON):
1) {"title": "string", "description": "string"}
2) {"text": "string"}

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

poetry install
poetry run pre-commit install
poetry run pre-commit run -a

## Data (DVC)

Данные лежат в data/ag_news и трекаются через DVC (data/ag_news.dvc).
Перед train/infer код пытается сделать dvc pull для data/ag_news.
Если данных нет (или dvc не настроен), датасет будет скачан из HuggingFace и сохранён локально.

## Logging (MLflow)

Поднять MLflow локально:
poetry run mlflow ui --host 127.0.0.1 --port 8081

Переопределение tracking uri через Hydra:
python -m poetry run python -m news_topic_classifier.commands train logging.tracking_uri=http://127.0.0.1:8081

## Train

python -m poetry run python -m news_topic_classifier.commands train
python -m poetry run python -m news_topic_classifier.commands train trainer.max_epochs=1 batch_size=32

Лучший чекпоинт: outputs/checkpoints/best.ckpt

## Infer

python -m poetry run python -m news_topic_classifier.commands infer

Параметры инференса (ckpt_path и тексты) задаются в configs/infer.yaml.
