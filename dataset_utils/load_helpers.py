from datasets import load_dataset, Dataset

def load_ru_news_parquet(path: str) -> Dataset:
    ds = load_dataset("parquet", data_files=path, split="train")
    # If columns "text" and "title" must be renamed
    if "text" in ds.column_names:
        ds = ds.rename_column("text", "passage")
    if "title" in ds.column_names:
        ds = ds.rename_column("title", "query")
    # Remove columns if they exist
    for col in ["url", "source", "timestamp"]:
        if col in ds.column_names:
            ds = ds.remove_columns(col)
    return ds

def _preprocess_ru_stackoverflow_multiple(batch: dict) -> dict:
    """
    Processes a batch of examples to handle multiple accepted answers.
    For each question, it creates a separate row for each of its accepted answers.
    """
    new_queries = []
    new_passages = []
    new_labels = []

    # The input `batch` is a dictionary of lists (e.g., batch['answers'] is a list of lists of answers)
    for i in range(len(batch["text_markdown"])):
        query = batch["text_markdown"][i]
        answers = batch["answers"][i]
        
        for answer in answers:
            if answer.get("is_accepted"):
                new_queries.append(query)
                new_passages.append(answer["text_markdown"])
                new_labels.append(1)

    return {
        "query": new_queries,
        "passage": new_passages,
        "label": new_labels
    }

def load_ru_stackoverflow_parquet(path: str, split: str = "train") -> Dataset:
    ds = load_dataset("parquet", data_files=path, split=split)

    # Process the dataset to handle multiple accepted answers

    ds = ds.map(
        _preprocess_ru_stackoverflow_multiple, 
        batched=True, 
        remove_columns=ds.column_names
    )
    return ds