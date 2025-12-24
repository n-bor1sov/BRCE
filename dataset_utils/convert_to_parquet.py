#!/usr/bin/env python3
"""Convert ru_news.jsonl.zst to parquet format."""
import argparse
import sys
from pathlib import Path
import zstandard as zstd
import io
import json
from typing import Generator

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset

def _iter_jsonl(path: str) -> Generator[dict, None, None]:
    with open(path, "r") as fh:
        for line in fh:
            yield json.loads(line)

def _iter_jsonl_zst(path: str) -> Generator[dict, None, None]:
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                yield json.loads(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert dataset to parquet format")
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default="data/ru_news/ru_news.jsonl.zst",
        help="Path to input .jsonl.zst file (default: data/ru_news/ru_news.jsonl.zst)",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["jsonl", "jsonl.zst"],
        nargs="?",
        default="jsonl.zst",
        help="Format of input file (default: jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output .parquet file (default: same directory as input with .parquet extension)",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: same directory as input, replace .zst with .parquet
        output_path = input_path.parent / input_path.stem.replace(".jsonl", "") + ".parquet"
    
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")
    
    print(f"Loading data from {input_path}...")
    if args.format == "jsonl.zst":
        ds = Dataset.from_generator(_iter_jsonl_zst, gen_kwargs={"path": str(input_path)})
    elif args.format == "jsonl":
        ds = Dataset.from_generator(_iter_jsonl, gen_kwargs={"path": str(input_path)})
    else:
        parser.error(f"Invalid format: {args.format}")

    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(output_path))
    
    print(f"Done! Converted {len(ds)} records to parquet format.")

