"""
Download and prepare WikiText-2 or WikiText-103 dataset for GPT2 training.

This script downloads WikiText dataset and converts it to the format
expected by the GPT2 training script:
- One sentence/line per line in the corpus files
- A vocabulary file with one token per line
"""

import os
import urllib.request
import argparse
from pathlib import Path
from collections import Counter
import re

# NEW: import datasets for WikiText-103
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def download_wikitext(dataset='wikitext-2', output_dir='data'):
    """Download WikiText dataset."""

    os.makedirs(output_dir, exist_ok=True)

    urls = {
        'wikitext-2': 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt',
        'wikitext-2-test': 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt',
        'wikitext-2-valid': 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt'
    }

    if dataset == 'wikitext-2':
        print(f"Downloading {dataset}...")

        train_path = os.path.join(output_dir, 'wiki.train.tokens')
        test_path = os.path.join(output_dir, 'wiki.test.tokens')
        valid_path = os.path.join(output_dir, 'wiki.valid.tokens')

        print(f"  Downloading training data...")
        urllib.request.urlretrieve(urls['wikitext-2'], train_path)

        print(f"  Downloading test data...")
        urllib.request.urlretrieve(urls['wikitext-2-test'], test_path)

        print(f"  Downloading validation data...")
        urllib.request.urlretrieve(urls['wikitext-2-valid'], valid_path)

        return output_dir

    elif dataset == 'wikitext-103':
        if load_dataset is None:
            raise ImportError(
                "HuggingFace 'datasets' library is required for WikiText-103. "
                "Install with: pip install datasets"
            )

        print(f"Downloading {dataset} using HuggingFace datasets...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")

        # Save splits to files in output_dir
        train_path = os.path.join(output_dir, 'wiki.train.tokens')
        test_path = os.path.join(output_dir, 'wiki.test.tokens')
        valid_path = os.path.join(output_dir, 'wiki.valid.tokens')

        ds['train'].to_pandas().to_csv(train_path, index=False, header=False)
        ds['test'].to_pandas().to_csv(test_path, index=False, header=False)
        ds['validation'].to_pandas().to_csv(valid_path, index=False, header=False)

        print("WikiText-103 downloaded and saved.")
        return output_dir

    else:
        raise ValueError(f"Dataset must be 'wikitext-2' or 'wikitext-103', got '{dataset}'")


def simple_tokenize(text):
    """Simple whitespace and punctuation tokenization."""
    text = re.sub(r'([.!?,;:])', r' \1 ', text)
    tokens = [t for t in text.split() if t.strip()]
    return tokens


def process_wikitext_file(input_path, output_path, min_length=5):
    """Process WikiText file to create corpus in the expected format."""
    print(f"Processing {input_path} -> {output_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_lines = []
    all_tokens = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('@'):
            continue

        tokens = simple_tokenize(line)
        if len(tokens) < min_length:
            continue

        processed_lines.append(' '.join(tokens))
        all_tokens.extend(tokens)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_lines))

    print(f"  Written {len(processed_lines)} lines to {output_path}")
    return all_tokens


def create_vocabulary(tokens, vocab_path, min_freq=2, max_vocab=50000):
    """Create vocabulary file from tokens."""
    print(f"Creating vocabulary at {vocab_path}")

    counter = Counter(tokens)
    vocab_items = [(token, count) for token, count in counter.items() if count >= min_freq]
    vocab_items.sort(key=lambda x: x[1], reverse=True)
    vocab_items = vocab_items[:max_vocab]

    vocab_tokens = ['<unk>'] + [token for token, _ in vocab_items]

    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab_tokens))

    print(f"  Created vocabulary with {len(vocab_tokens)} tokens")
    return vocab_tokens


def main():
    parser = argparse.ArgumentParser(description='Download and prepare WikiText dataset for GPT2')
    parser.add_argument('--dataset', type=str, default='wikitext-2',
                        choices=['wikitext-2', 'wikitext-103'],
                        help='WikiText dataset to download')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory to download raw data')
    parser.add_argument('--build-dir', type=str, default='build',
                        help='Directory to store processed data')
    parser.add_argument('--min-freq', type=int, default=2,
                        help='Minimum token frequency for vocabulary')
    parser.add_argument('--max-vocab', type=int, default=50000,
                        help='Maximum vocabulary size')

    args = parser.parse_args()

    os.makedirs(args.build_dir, exist_ok=True)

    data_dir = download_wikitext(args.dataset, args.data_dir)

    train_file = os.path.join(data_dir, 'wiki.train.tokens')
    valid_file = os.path.join(data_dir, 'wiki.valid.tokens')
    test_file = os.path.join(data_dir, 'wiki.test.tokens')

    print("\nProcessing WikiText files...\n")

    train_tokens = process_wikitext_file(train_file, os.path.join(args.build_dir, 'corpus.train.txt'))
    test_tokens = process_wikitext_file(test_file, os.path.join(args.build_dir, 'corpus.test.txt'))

    print("\nCreating vocabulary...\n")
    vocab = create_vocabulary(train_tokens, os.path.join(args.build_dir, 'vocab.txt'),
                              min_freq=args.min_freq, max_vocab=args.max_vocab)

    print("\nDataset preparation complete!")
    print(f"Files created in '{args.build_dir}/':")
    print(f"  - corpus.train.txt  ({len(train_tokens)} tokens)")
    print(f"  - corpus.test.txt   ({len(test_tokens)} tokens)")
    print(f"  - vocab.txt         ({len(vocab)} tokens)")


if __name__ == '__main__':
    main()
