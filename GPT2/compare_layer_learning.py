"""
Direct Comparison: Is Layer 12 learning as effectively as Layer 1?

We found that:
- Layer 12 has 5.1x amplification from power
- But also has 23x worse SNR

Which effect dominates? Let's directly compare learning efficiency.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from src.gpt2.modeling import Transformer
from src.gpt2.data import Vocab, TokenizedCorpus

print("="*80)
print("DIRECT LAYER-BY-LAYER LEARNING COMPARISON")
print("="*80)
print()

# Configuration
SEQ_LEN = 128
LAYERS = 12
HEADS = 12
DIMS = 576
RATE = 4
DROPOUT = 0.0  # No dropout for deterministic testing

# Load vocabulary
print("Loading vocabulary...")
vocab = Vocab(vocab_path='GPT2/build/vocab.txt')

# Create model
print("Creating model...")
model = Transformer(
    layers=LAYERS,
    pad_idx=vocab.pad_idx,
    words=len(vocab),
    seq_len=SEQ_LEN,
    heads=HEADS,
    dims=DIMS,
    rate=RATE,
    dropout=DROPOUT,
    bidirectional=False
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Load checkpoint
print("Loading checkpoint...")
ckpt = torch.load('GPT2/best-no-relu-3.8.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])

# Get a sample batch
print("Loading sample batch...")
eval_data = TokenizedCorpus('GPT2/build/corpus.test.txt', vocab, SEQ_LEN)
data = eval_data.fetch(32)
input_ids = data['input'].to(device)
output_ids = data['output'].to(device)

print()
print("="*80)
print("STEP 1: BASELINE FORWARD PASS")
print("="*80)
print()

# Get baseline loss
model.eval()
with torch.no_grad():
    output = model(input_ids, None)
    logits_before = output[0] if isinstance(output, tuple) else output
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, reduction='mean')
    loss_before = criterion(logits_before.transpose(1, 2), output_ids)

print(f"Baseline loss: {loss_before.item():.6f}")
print()

print("="*80)
print("STEP 2: COMPUTE GRADIENTS FOR ALL LAYERS")
print("="*80)
print()

model.train()

# Forward and backward to get gradients
output = model(input_ids, None)
logits = output[0] if isinstance(output, tuple) else output
loss = criterion(logits.transpose(1, 2), output_ids)
loss.backward()

# Store gradients for all layers
layer_gradients = {}
for i in range(12):
    layer_num = i + 1
    layer = model.transformers[i]

    layer_gradients[layer_num] = {
        'linear1_weight_grad': layer.ff.linear1.weight.grad.clone(),
        'linear2_weight_grad': layer.ff.linear2.weight.grad.clone(),
        'linear1_weight': layer.ff.linear1.weight.data.clone(),
        'linear2_weight': layer.ff.linear2.weight.data.clone(),
    }

print("Gradients computed for all 12 layers")
print()

print("="*80)
print("STEP 3: UPDATE EACH LAYER INDIVIDUALLY AND MEASURE IMPACT")
print("="*80)
print()

learning_rate = 1e-4
results = {}

for layer_num in range(1, 13):
    print(f"Testing Layer {layer_num}...")

    # Reset model to original state
    model.load_state_dict(ckpt['model'])

    # Apply update ONLY to this layer
    with torch.no_grad():
        layer = model.transformers[layer_num - 1]
        grad_w1 = layer_gradients[layer_num]['linear1_weight_grad']
        grad_w2 = layer_gradients[layer_num]['linear2_weight_grad']

        layer.ff.linear1.weight.data -= learning_rate * grad_w1
        layer.ff.linear2.weight.data -= learning_rate * grad_w2

    # Measure new loss
    model.eval()
    with torch.no_grad():
        output = model(input_ids, None)
        logits_after = output[0] if isinstance(output, tuple) else output
        loss_after = criterion(logits_after.transpose(1, 2), output_ids)

    # Compute impact
    delta_loss = loss_after.item() - loss_before.item()

    results[layer_num] = {
        'loss_before': loss_before.item(),
        'loss_after': loss_after.item(),
        'delta_loss': delta_loss,
        'grad_norm_w1': grad_w1.norm().item(),
        'grad_norm_w2': grad_w2.norm().item(),
    }

    print(f"  Loss change: {delta_loss:+.6f}")

print()

print("="*80)
print("STEP 4: RESULTS - LEARNING EFFICIENCY BY LAYER")
print("="*80)
print()

print(f"{'Layer':<8} {'Grad Norm':<15} {'Δ Loss':<15} {'Efficiency':<15} {'Notes':<20}")
print("-"*80)

# Calculate efficiency = |Δloss| / grad_norm (how much loss changes per unit gradient)
for layer_num in range(1, 13):
    r = results[layer_num]
    grad_norm = (r['grad_norm_w1'] + r['grad_norm_w2']) / 2  # Average
    delta_loss = r['delta_loss']
    efficiency = abs(delta_loss) / grad_norm if grad_norm > 0 else 0

    # Determine if loss improved
    if delta_loss < 0:
        note = "✅ Improved"
    elif delta_loss > 0:
        note = "❌ Worsened"
    else:
        note = "- No change"

    print(f"{layer_num:<8} {grad_norm:<15.6e} {delta_loss:<+15.6e} {efficiency:<15.6e} {note:<20}")

print()

print("="*80)
print("STEP 5: COMPARISON - EARLY VS LATE LAYERS")
print("="*80)
print()

# Compare layer groups
early_layers = [1, 2, 3]
middle_layers = [5, 6, 7]
late_layers = [10, 11, 12]

def avg_efficiency(layer_nums):
    efficiencies = []
    for l in layer_nums:
        r = results[l]
        grad_norm = (r['grad_norm_w1'] + r['grad_norm_w2']) / 2
        efficiency = abs(r['delta_loss']) / grad_norm if grad_norm > 0 else 0
        efficiencies.append(efficiency)
    return sum(efficiencies) / len(efficiencies)

early_eff = avg_efficiency(early_layers)
middle_eff = avg_efficiency(middle_layers)
late_eff = avg_efficiency(late_layers)

print(f"Average Learning Efficiency:")
print(f"  Early layers (1-3):   {early_eff:.6e}")
print(f"  Middle layers (5-7):  {middle_eff:.6e}")
print(f"  Late layers (10-12):  {late_eff:.6e}")
print()

print(f"Relative to Early Layers:")
print(f"  Middle: {middle_eff/early_eff:.2f}x")
print(f"  Late:   {late_eff/early_eff:.2f}x")
print()

print("="*80)
print("STEP 6: POWER ACTIVATION CONTRIBUTION")
print("="*80)
print()

# Check power values
print(f"{'Layer':<8} {'Power':<10} {'Activation Max':<15} {'Learning Efficiency':<20}")
print("-"*80)

for layer_num in range(1, 13):
    power = model.transformers[layer_num-1].ff.activation.power
    r = results[layer_num]
    grad_norm = (r['grad_norm_w1'] + r['grad_norm_w2']) / 2
    efficiency = abs(r['delta_loss']) / grad_norm if grad_norm > 0 else 0

    # Get activation magnitude (from checkpoint info)
    # We'll estimate from gradient norm as proxy
    print(f"{layer_num:<8} {power:<10} {'N/A':<15} {efficiency:<20.6e}")

print()

print("="*80)
print("STEP 7: DETAILED ANALYSIS - WHY THE DIFFERENCES?")
print("="*80)
print()

# Find best and worst performing layers
sorted_layers = sorted(results.items(),
                       key=lambda x: abs(x[1]['delta_loss']) /
                                    ((x[1]['grad_norm_w1'] + x[1]['grad_norm_w2'])/2),
                       reverse=True)

best_layer = sorted_layers[0][0]
worst_layer = sorted_layers[-1][0]

print(f"Most effective layer: Layer {best_layer}")
r = results[best_layer]
grad_norm = (r['grad_norm_w1'] + r['grad_norm_w2']) / 2
eff = abs(r['delta_loss']) / grad_norm
print(f"  Gradient norm: {grad_norm:.6e}")
print(f"  Loss change: {r['delta_loss']:+.6e}")
print(f"  Efficiency: {eff:.6e}")
print(f"  Power: {model.transformers[best_layer-1].ff.activation.power}")
print()

print(f"Least effective layer: Layer {worst_layer}")
r = results[worst_layer]
grad_norm = (r['grad_norm_w1'] + r['grad_norm_w2']) / 2
eff = abs(r['delta_loss']) / grad_norm
print(f"  Gradient norm: {grad_norm:.6e}")
print(f"  Loss change: {r['delta_loss']:+.6e}")
print(f"  Efficiency: {eff:.6e}")
print(f"  Power: {model.transformers[worst_layer-1].ff.activation.power}")
print()

ratio = (abs(results[best_layer]['delta_loss']) / ((results[best_layer]['grad_norm_w1'] + results[best_layer]['grad_norm_w2'])/2)) / \
        (abs(results[worst_layer]['delta_loss']) / ((results[worst_layer]['grad_norm_w1'] + results[worst_layer]['grad_norm_w2'])/2))
print(f"Best performs {ratio:.2f}x better than worst")
print()

print("="*80)
print("FINAL ANSWER")
print("="*80)
print()

print("Q: Is Layer 12 learning as effectively as Layer 1?")
print()

layer1_eff = abs(results[1]['delta_loss']) / ((results[1]['grad_norm_w1'] + results[1]['grad_norm_w2'])/2)
layer12_eff = abs(results[12]['delta_loss']) / ((results[12]['grad_norm_w1'] + results[12]['grad_norm_w2'])/2)
ratio_12_to_1 = layer12_eff / layer1_eff

print(f"Layer 1 learning efficiency:  {layer1_eff:.6e}")
print(f"Layer 12 learning efficiency: {layer12_eff:.6e}")
print()
print(f"Ratio (L12 / L1): {ratio_12_to_1:.3f}")
print()

if ratio_12_to_1 > 1.2:
    print("✅ YES! Layer 12 learns MORE effectively than Layer 1!")
    print(f"   Layer 12 is {ratio_12_to_1:.2f}x more efficient")
    print()
    print("   Why? Power activation amplification > SNR degradation")
elif ratio_12_to_1 > 0.8:
    print("✅ YES! Layer 12 learns about AS effectively as Layer 1")
    print(f"   Layer 12 is {ratio_12_to_1:.2f}x as efficient")
    print()
    print("   The power activation successfully compensates for SNR issues")
elif ratio_12_to_1 > 0.3:
    print("🟡 PARTIALLY. Layer 12 learns less effectively than Layer 1")
    print(f"   Layer 12 is {ratio_12_to_1:.2f}x as efficient")
    print()
    print("   Power helps but doesn't fully compensate")
else:
    print("❌ NO. Layer 12 learns much less effectively than Layer 1")
    print(f"   Layer 12 is only {ratio_12_to_1:.2f}x as efficient")
    print()
    print("   Severe learning impairment despite power amplification")

print()

print("Key insights:")
print(f"  1. Early layer efficiency:  {early_eff:.6e}")
print(f"  2. Late layer efficiency:   {late_eff:.6e}")
print(f"  3. Ratio:                   {late_eff/early_eff:.3f}x")
print()

if late_eff > early_eff:
    print("  🎉 LATE LAYERS LEARN BETTER!")
    print("     The power activation not only compensates but OVERCOMPENSATES")
    print("     for the SNR degradation, making later layers more effective!")
elif late_eff > 0.7 * early_eff:
    print("  ✅ LAYERS LEARN ROUGHLY EQUALLY")
    print("     The power activation successfully balances learning across depths")
elif late_eff > 0.3 * early_eff:
    print("  🟡 LATE LAYERS LEARN SLOWER")
    print("     Power helps but doesn't fully compensate for SNR degradation")
else:
    print("  ❌ LATE LAYERS SEVERELY IMPAIRED")
    print("     Despite power amplification, late layers struggle to learn")

print()
print("="*80)
