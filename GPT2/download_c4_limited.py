"""
Download and prepare C4 dataset (limited to ~20GB) for GPT2 training with GPT-2 tokenizer.
C4 provides more diverse text than Wikipedia for better benchmark performance.
"""

import os
import argparse
import gc
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import islice
import warnings

# Suppress tokenizer warnings about sequence length
warnings.filterwarnings('ignore', message='.*sequence length is longer than.*')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


# Global tokenizer - loaded once per process
_tokenizer = None

def get_tokenizer():
    """Get or create tokenizer (one per process)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return _tokenizer

def process_example(example, seq_len=128):
    """Process a single example - designed to be called in parallel."""
    tokenizer = get_tokenizer()

    text = example['text'].strip()
    if not text:
        return []

    # Tokenize with GPT-2
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    sequences = []
    # Split into chunks of seq_len to avoid truncation during training
    for i in range(0, len(token_ids), seq_len - 2):  # -2 for BOS/EOS
        chunk = token_ids[i:i + seq_len - 2]
        if len(chunk) < 5:  # Skip very short sequences
            continue

        # OPTIMIZED: Decode the entire chunk at once
        decoded_chunk = tokenizer.decode(chunk)
        tokens = decoded_chunk.split()
        sequences.append(' '.join(tokens))

    return sequences


def download_and_tokenize_c4(output_dir='build', seq_len=64, num_workers=None, max_train_docs=25_000_000, max_test_docs=64_000):
    """
    Download C4 dataset (limited size) and create pre-tokenized corpus files.

    Args:
        output_dir: Directory to save processed files
        seq_len: Sequence length for chunking
        num_workers: Number of CPU cores to use
        max_train_docs: Maximum training documents (~25M docs ≈ 20GB)
        max_test_docs: Maximum test documents

    Note: C4 is more diverse than Wikipedia - includes web pages, blogs, forums, etc.
    Better for benchmark performance on varied downstream tasks.
    """

    if num_workers is None:
        num_workers = cpu_count()

    print(f"Using {num_workers} CPU cores for parallel processing")
    print("Loading C4 dataset in streaming mode (memory efficient)...")
    print(f"Target: ~{max_train_docs:,} training documents (~20-25GB)")
    print("\nC4 Dataset Info:")
    print("  - Colossal Clean Crawled Corpus from Common Crawl")
    print("  - More diverse than Wikipedia (web pages, blogs, forums, etc.)")
    print("  - Heavily filtered for quality and safety")
    print("  - Better generalization for downstream benchmarks")

    # Load dataset in streaming mode to avoid loading entire dataset into memory
    # Using 'en' (English) subset of C4
    dataset = load_dataset('allenai/c4', 'en', split='train', streaming=True)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'corpus.train.txt')
    test_path = os.path.join(output_dir, 'corpus.test.txt')

    # Process test data FIRST (faster, so we can catch errors early)
    if os.path.exists(test_path):
        print(f"\nTest file already exists: {test_path}")
        print("Skipping test data processing...")
    else:
        print(f"\nProcessing test data (first {max_test_docs:,} documents)...")
        # Take first N documents for test set
        test_dataset = islice(dataset, max_test_docs)
        process_split_parallel(test_dataset, test_path, seq_len, num_workers)

    # Process training data SECOND (slower, but now we know the pipeline works)
    if os.path.exists(train_path):
        print(f"\nTraining file already exists: {train_path}")
        print("Skipping training data processing...")
    else:
        print(f"\nProcessing training data ({max_train_docs:,} documents after test data)...")
        # Reload dataset since iterator was consumed by test data
        dataset_train = load_dataset('allenai/c4', 'en', split='train', streaming=True)
        # Skip test documents, then take training documents
        train_dataset = islice(dataset_train, max_test_docs, max_test_docs + max_train_docs)
        process_split_parallel(train_dataset, train_path, seq_len, num_workers)

    print("\nDataset preparation complete!")
    print(f"Files created in '{output_dir}/':")
    print(f"  - corpus.train.txt")
    print(f"  - corpus.test.txt")
    print(f"  - NO vocab.txt needed (using GPT-2 pretrained vocabulary)")


def process_split_parallel(dataset, output_path, seq_len, num_workers):
    """Process dataset split using parallel processing."""

    # Create partial function with seq_len bound
    process_fn = partial(process_example, seq_len=seq_len)

    with open(output_path, 'w', encoding='utf-8', buffering=8192) as f:
        sequence_count = 0
        batch_count = 0

        # Process in parallel with a pool
        with Pool(num_workers) as pool:
            # Use imap_unordered for streaming results (memory efficient, no total to avoid loading dataset)
            for sequences in tqdm(pool.imap_unordered(process_fn, dataset, chunksize=100),
                                 desc="Processing"):
                for seq in sequences:
                    f.write(seq + '\n')
                    sequence_count += 1

                # Periodically flush and collect garbage to prevent memory buildup
                batch_count += 1
                if batch_count % 1000 == 0:
                    f.flush()
                    gc.collect()

    print(f"  Written {sequence_count:,} sequences to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Download and prepare C4 dataset (limited size)')
    parser.add_argument('--output-dir', type=str, default='build',
                        help='Directory to store processed data')
    parser.add_argument('--seq-len', type=int, default=1024,
                        help='Sequence length for chunking')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU cores to use (default: all available)')
    parser.add_argument('--max-train-docs', type=int, default=5_000_000,
                        help='Maximum training documents (~25M ≈ 20GB)')
    parser.add_argument('--max-test-docs', type=int, default=64_000,
                        help='Maximum test documents')

    args = parser.parse_args()

    download_and_tokenize_c4(
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        max_train_docs=args.max_train_docs,
        max_test_docs=args.max_test_docs
    )


if __name__ == '__main__':
    main()
