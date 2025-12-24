### RU NEWS ###
HF_HUB_DISABLE_XET=1 uv run hf download --repo-type dataset IlyaGusev/ru_news --local-dir data/ru_news

# Convert zst to parquet if not already converted
if [ ! -f data/ru_news/ru_news.parquet ]; then
    uv run python -m dataset_utils.convert_to_parquet data/ru_news/ru_news.jsonl.zst -o data/ru_news/ru_news.parquet
fi

### RU STACKOVERFLOW ###
HF_HUB_DISABLE_XET=1 uv run hf download --repo-type dataset IlyaGusev/ru_stackoverflow --local-dir data/ru_stackoverflow

if [ ! -f data/ru_stackoverflow/ru_stackoverflow.parquet ]; then
    uv run python -m dataset_utils.convert_to_parquet data/ru_stackoverflow/ru_stackoverflow.jsonl.zst -o data/ru_stackoverflow/ru_stackoverflow.parquet
fi