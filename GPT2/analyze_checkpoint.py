"""
Analysis script for ProperNeuronLLM checkpoint

Checks:
1. Next-word prediction accuracy
2. Vector collapse detection
3. Embedding diversity metrics
4. Model quality assessment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from collections import defaultdict
from pathlib import Path


# ============================================================================
# LOAD MODEL ARCHITECTURE
# ============================================================================

class ProperNeuronLLM_Fast(nn.Module):
    """Same architecture as training file"""
    def __init__(self, seq_len=125, vocab_size=25000, embed_dim=160):
        super().__init__()

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.input_scalars = nn.Parameter(
            torch.randn(seq_len, vocab_size) * 0.01
        )

        self.weight_vectors = nn.Parameter(
            torch.randn(seq_len, vocab_size, embed_dim) * 0.01
        )

        self.output_embeddings = nn.Parameter(
            torch.randn(seq_len, vocab_size, embed_dim) * 0.01
        )

        self.query_proj = nn.Parameter(
            torch.randn(embed_dim, embed_dim) * 0.01
        )
        self.key_proj = nn.Parameter(
            torch.randn(embed_dim, embed_dim) * 0.01
        )

    def forward_batch(self, input_tokens):
        """Process ALL positions at once"""
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        if seq_len > self.seq_len:
            input_tokens = input_tokens[:, :self.seq_len]
            seq_len = self.seq_len

        pos_idx = torch.arange(seq_len, device=device)
        pos_idx = pos_idx.unsqueeze(0).expand(batch_size, -1)

        # Gather activations
        input_activations = self.input_scalars[pos_idx, input_tokens]
        weight_vecs = self.weight_vectors[pos_idx, input_tokens]

        # Compute neuron outputs
        neuron_outputs = input_activations.unsqueeze(-1) * weight_vecs

        final_outputs = neuron_outputs

        # Cumulative sum for context
        context_vectors = torch.cumsum(final_outputs, dim=1)

        # Shift by 1
        context_vectors = torch.cat([
            torch.zeros(batch_size, 1, self.embed_dim, device=device),
            context_vectors[:, :-1, :]
        ], dim=1)

        # Compute logits for all positions
        logits_list = []
        for pos in range(seq_len):
            context = context_vectors[:, pos, :]
            pos_embeddings = self.output_embeddings[pos]

            context_norm = F.normalize(context, dim=1, eps=1e-8)
            embeddings_norm = F.normalize(pos_embeddings, dim=1, eps=1e-8)

            logits_pos = context_norm @ embeddings_norm.T * 10.0
            logits_list.append(logits_pos)

        logits = torch.stack(logits_list, dim=1)

        return logits


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_checkpoint(checkpoint_path, device='cpu'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    print(f"Config: seq_len={config['seq_len']}, vocab_size={config['vocab_size']}, embed_dim={config['embed_dim']}")

    model = ProperNeuronLLM_Fast(
        seq_len=config['seq_len'],
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("Checkpoint loaded successfully!")
    return model, config


def check_vector_collapse(model):
    """
    Check if output embeddings have collapsed (all similar)

    Returns:
        dict with collapse metrics
    """
    print("\n" + "="*70)
    print("CHECKING FOR VECTOR COLLAPSE")
    print("="*70)

    results = {}

    # Check output embeddings at different positions
    for pos_idx in [0, model.seq_len // 4, model.seq_len // 2, 3 * model.seq_len // 4, model.seq_len - 1]:
        print(f"\nPosition {pos_idx}:")

        embeddings = model.output_embeddings[pos_idx].detach().cpu()  # (vocab_size, embed_dim)

        # 1. Check norms
        norms = torch.norm(embeddings, dim=1)
        print(f"  Norms - Mean: {norms.mean():.4f}, Std: {norms.std():.4f}, Min: {norms.min():.4f}, Max: {norms.max():.4f}")

        # 2. Check if all vectors are similar (cosine similarity)
        normalized = F.normalize(embeddings, dim=1, eps=1e-8)

        # Sample 1000 random pairs to check similarity
        num_samples = min(1000, model.vocab_size)
        indices = torch.randperm(model.vocab_size)[:num_samples]
        sample_vecs = normalized[indices]

        # Compute pairwise cosine similarities
        similarity_matrix = sample_vecs @ sample_vecs.T

        # Get off-diagonal elements (don't compare vector to itself)
        mask = ~torch.eye(num_samples, dtype=bool)
        off_diag_sims = similarity_matrix[mask]

        mean_sim = off_diag_sims.mean().item()
        std_sim = off_diag_sims.std().item()
        max_sim = off_diag_sims.max().item()
        min_sim = off_diag_sims.min().item()

        print(f"  Pairwise Cosine Similarity - Mean: {mean_sim:.4f}, Std: {std_sim:.4f}")
        print(f"  Pairwise Cosine Similarity - Min: {min_sim:.4f}, Max: {max_sim:.4f}")

        # 3. Check diversity via variance
        per_dim_variance = embeddings.var(dim=0).mean().item()
        print(f"  Per-dimension variance (avg): {per_dim_variance:.4f}")

        # Collapse detection
        collapse_warning = False
        if mean_sim > 0.8:
            print(f"  WARNING: High mean similarity ({mean_sim:.4f}) - possible collapse!")
            collapse_warning = True
        if std_sim < 0.05:
            print(f"  WARNING: Low similarity variance ({std_sim:.4f}) - vectors too uniform!")
            collapse_warning = True
        if per_dim_variance < 0.001:
            print(f"  WARNING: Low per-dimension variance ({per_dim_variance:.4f}) - possible collapse!")
            collapse_warning = True

        if not collapse_warning:
            print(f"  [OK] No collapse detected at position {pos_idx}")

        results[f'pos_{pos_idx}'] = {
            'norm_mean': norms.mean().item(),
            'norm_std': norms.std().item(),
            'similarity_mean': mean_sim,
            'similarity_std': std_sim,
            'per_dim_variance': per_dim_variance,
            'collapsed': collapse_warning
        }

    return results


def evaluate_next_word_prediction(model, tokenizer, device='cpu', num_samples=500, max_length=64, training_max_length=64):
    """
    Evaluate next-word prediction accuracy on validation set

    Returns:
        dict with accuracy metrics
    """
    print("\n" + "="*70)
    print("EVALUATING NEXT-WORD PREDICTION ACCURACY")
    print("="*70)

    # Load validation dataset
    print("\nLoading WikiText-103 validation set...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')

    # Prepare sequences
    sequences = []
    for item in dataset:
        text = item['text']
        if len(text.strip()) == 0:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) < 5:
            continue

        # Take sequences of varying lengths
        for i in range(0, len(tokens) - 5, max_length // 2):
            seq_len = min(max_length, len(tokens) - i)
            if seq_len >= 5:
                sequences.append(tokens[i:i + seq_len])

        if len(sequences) >= num_samples:
            break

    sequences = sequences[:num_samples]
    print(f"Evaluating on {len(sequences)} sequences")

    # Evaluation metrics
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    total_predictions = 0

    position_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})

    model.eval()
    with torch.no_grad():
        for seq_idx, sequence in enumerate(sequences):
            if (seq_idx + 1) % 100 == 0:
                print(f"  Processed {seq_idx + 1}/{len(sequences)} sequences...")

            # Split into input and target
            # IMPORTANT: Truncate to training_max_length to match training distribution
            seq_truncated = sequence[:training_max_length]
            input_tokens = torch.tensor(seq_truncated[:-1]).unsqueeze(0).to(device)  # (1, seq_len-1)
            target_tokens = torch.tensor(seq_truncated[1:]).to(device)  # (seq_len-1,)

            # Get predictions
            logits = model.forward_batch(input_tokens)  # (1, seq_len-1, vocab_size)
            logits = logits.squeeze(0)  # (seq_len-1, vocab_size)

            # Check each position (skip first 4 positions - model trained with min_length=5)
            # Position 0-3 have insufficient context (< 4 tokens)
            start_pos = 4  # Start evaluating from position 4 (5th token, has 4 tokens of context)

            for pos in range(start_pos, len(target_tokens)):
                target_token = target_tokens[pos].item()
                pos_logits = logits[pos]  # (vocab_size,)

                # Get top-k predictions
                top10_preds = torch.topk(pos_logits, k=10).indices.cpu().tolist()

                # Check if target is in top-k
                if target_token == top10_preds[0]:
                    top1_correct += 1
                    top5_correct += 1
                    top10_correct += 1
                    position_accuracy[pos]['correct'] += 1
                elif target_token in top10_preds[:5]:
                    top5_correct += 1
                    top10_correct += 1
                elif target_token in top10_preds:
                    top10_correct += 1

                position_accuracy[pos]['total'] += 1
                total_predictions += 1

    # Calculate accuracies
    top1_acc = top1_correct / total_predictions if total_predictions > 0 else 0
    top5_acc = top5_correct / total_predictions if total_predictions > 0 else 0
    top10_acc = top10_correct / total_predictions if total_predictions > 0 else 0

    print(f"\n{'='*70}")
    print(f"ACCURACY RESULTS (positions 4-{training_max_length-1}, matching training)")
    print(f"{'='*70}")
    print(f"Total predictions: {total_predictions}")
    print(f"Top-1 Accuracy:  {top1_acc*100:.2f}% ({top1_correct}/{total_predictions})")
    print(f"Top-5 Accuracy:  {top5_acc*100:.2f}% ({top5_correct}/{total_predictions})")
    print(f"Top-10 Accuracy: {top10_acc*100:.2f}% ({top10_correct}/{total_predictions})")

    # Show accuracy by position
    print(f"\nAccuracy by position (positions 4-13, with context >= 4 tokens):")
    for pos in sorted(position_accuracy.keys())[:10]:
        stats = position_accuracy[pos]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  Position {pos}: {acc*100:.2f}% ({stats['correct']}/{stats['total']})")

    return {
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'top10_accuracy': top10_acc,
        'total_predictions': total_predictions,
        'position_accuracy': dict(position_accuracy)
    }


def analyze_embedding_diversity(model):
    """
    Analyze diversity of embeddings across vocabulary
    """
    print("\n" + "="*70)
    print("ANALYZING EMBEDDING DIVERSITY")
    print("="*70)

    # Check a few positions
    positions_to_check = [0, model.seq_len // 2, model.seq_len - 1]

    for pos in positions_to_check:
        print(f"\nPosition {pos}:")
        embeddings = model.output_embeddings[pos].detach().cpu()  # (vocab_size, embed_dim)

        # Normalize
        normalized = F.normalize(embeddings, dim=1, eps=1e-8)

        # Principal component analysis (rough approximation)
        # Compute covariance matrix
        mean_vec = embeddings.mean(dim=0)
        centered = embeddings - mean_vec

        # Singular values give variance along principal components
        U, S, V = torch.svd(centered)

        # Check how much variance is captured by top components
        total_var = (S ** 2).sum()
        top5_var = (S[:5] ** 2).sum() / total_var
        top10_var = (S[:10] ** 2).sum() / total_var

        print(f"  Variance captured by top 5 components: {top5_var*100:.2f}%")
        print(f"  Variance captured by top 10 components: {top10_var*100:.2f}%")

        # Check for clusters (random sample)
        sample_size = min(100, model.vocab_size)
        indices = torch.randperm(model.vocab_size)[:sample_size]
        sample = normalized[indices]

        # Compute nearest neighbor distances
        distances = []
        for i in range(sample_size):
            vec = sample[i:i+1]
            sims = (vec @ sample.T).squeeze()
            sims[i] = -1  # Ignore self
            max_sim = sims.max().item()
            distances.append(1 - max_sim)  # Convert to distance

        avg_nearest_dist = np.mean(distances)
        print(f"  Average nearest-neighbor distance: {avg_nearest_dist:.4f}")

        if avg_nearest_dist < 0.1:
            print(f"  WARNING: Embeddings are very clustered!")
        else:
            print(f"  [OK] Good diversity - embeddings are well-separated")


def sample_predictions(model, tokenizer, device='cpu'):
    """
    Show some example predictions
    """
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)

    test_sentences = [
        "The cat sat on the",
        "Hello world, how are",
        "Machine learning is a",
        "The quick brown fox",
    ]

    model.eval()
    with torch.no_grad():
        for sentence in test_sentences:
            tokens = tokenizer.encode(sentence, add_special_tokens=True)
            input_tokens = torch.tensor(tokens).unsqueeze(0).to(device)

            logits = model.forward_batch(input_tokens)
            last_logits = logits[0, -1, :]  # Last position predictions

            # Get top 5 predictions
            top5 = torch.topk(last_logits, k=5)

            print(f"\nInput: '{sentence}'")
            print(f"Top 5 predictions:")
            for i, (score, idx) in enumerate(zip(top5.values, top5.indices)):
                token = tokenizer.decode([idx.item()])
                # Handle unicode encoding issues
                try:
                    print(f"  {i+1}. '{token}' (score: {score.item():.3f})")
                except UnicodeEncodeError:
                    print(f"  {i+1}. [token_{idx.item()}] (score: {score.item():.3f})")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    checkpoint_path = 'proper_neuron_llm_FAST_checkpoint.pt'

    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train_proper_neuron_llm_FAST.py")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model, config = load_checkpoint(checkpoint_path, device)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Run analyses
    print("\n" + "="*70)
    print("STARTING COMPREHENSIVE ANALYSIS")
    print("="*70)

    # 1. Check for vector collapse
    collapse_results = check_vector_collapse(model)

    # 2. Analyze embedding diversity
    analyze_embedding_diversity(model)

    # 3. Evaluate prediction accuracy
    # Use max_length=64 to match training (not seq_len=125)
    accuracy_results = evaluate_next_word_prediction(
        model, tokenizer, device,
        num_samples=500,  # Adjust based on how thorough you want
        max_length=64,  # Match training max_length
        training_max_length=64  # Explicit parameter
    )

    # 4. Sample predictions
    sample_predictions(model, tokenizer, device)

    # Summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)

    # Check if any collapse detected
    any_collapse = any(v['collapsed'] for v in collapse_results.values())
    if any_collapse:
        print("[X] VECTOR COLLAPSE DETECTED - model may need different loss or regularization")
    else:
        print("[OK] NO VECTOR COLLAPSE - embeddings are diverse")

    print(f"\nPrediction Performance:")
    print(f"  Top-1 Accuracy: {accuracy_results['top1_accuracy']*100:.2f}%")
    print(f"  Top-5 Accuracy: {accuracy_results['top5_accuracy']*100:.2f}%")

    # Interpretation
    top1 = accuracy_results['top1_accuracy']
    if top1 > 0.05:
        print(f"  [OK] Model is learning! (Random baseline ~0.002% for 50K vocab)")
    elif top1 > 0.001:
        print(f"  [!] Model is learning but slowly")
    else:
        print(f"  [X] Model barely better than random")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
