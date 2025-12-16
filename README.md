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

## Train
(будет добавлено в Task 2)
poetry run python -m news_topic_classifier.commands train
