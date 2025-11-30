"""
Gradient logging utility for monitoring Power-ReLU training dynamics.
"""

import torch
import torch.nn as nn


def log_gradient_stats(model, step: int, verbose: bool = False):
    """
    Log gradient and activation statistics for all transformer layers.

    Args:
        model: The Transformer model
        step: Current training step
        verbose: If True, print detailed per-layer statistics
    """
    print(f"\n{'='*80}")
    print(f"GRADIENT ANALYSIS - Step {step}")
    print(f"{'='*80}")

    # Header
    print(f"{'Layer':<6} {'Power':<6} {'Pre-Power':<35} {'Post-Power/ReLU':<35}")
    print(f"{'':6} {'':6} {'Grad Norm':<12} {'Activation':<22} {'Grad Norm':<12} {'Activation':<22}")
    print(f"{'-'*80}")

    for i, transformer_layer in enumerate(model.transformers):
        ff = transformer_layer.ff
        stats = ff.get_gradient_stats()

        # Extract key metrics
        power = stats['power']

        # Pre-power gradients (gradient flowing into the power operation)
        pre_grad = stats['gradients'].get('pre_power', {})
        pre_grad_norm = pre_grad.get('norm', 0)

        # Post-power gradients (gradient after power transformation)
        post_grad = stats['gradients'].get('post_power', {})
        post_grad_norm = post_grad.get('norm', 0)

        # Activations
        pre_power = stats['forward'].get('pre_power', {})
        pre_mean = pre_power.get('mean', 0)
        pre_std = pre_power.get('std', 0)

        post_power = stats['forward'].get('post_power', {})
        post_mean = post_power.get('mean', 0)
        post_std = post_power.get('std', 0)

        post_relu = stats['forward'].get('post_relu', {})
        relu_zeros_pct = post_relu.get('zeros_pct', 0)

        # Post-norm stats (if available)
        post_norm = stats['forward'].get('post_norm', {})

        # Weight gradients
        w1_grad = stats['weight_grads']['linear1']
        w2_grad = stats['weight_grads']['linear2']

        # Compact display
        pre_act_str = f"μ={pre_mean:+.3f} σ={pre_std:.3f}"
        post_act_str = f"μ={post_mean:+.3f} σ={post_std:.3f}"

        print(f"{i+1:<6} {power:<6} "
              f"{pre_grad_norm:<12.6f} {pre_act_str:<22} "
              f"{post_grad_norm:<12.6f} {post_act_str:<22}")

        # Verbose mode - show detailed breakdown
        if verbose:
            print(f"       {'':6} Weight Grads: linear1={w1_grad:.6f}, linear2={w2_grad:.6f}")
            print(f"       {'':6} ReLU zeros: {relu_zeros_pct:.1f}%")

            if pre_grad:
                print(f"       {'':6} Pre-power grad:  mean={pre_grad.get('mean', 0):+.6f}, "
                      f"std={pre_grad.get('std', 0):.6f}, max_abs={pre_grad.get('max_abs', 0):.6f}")

            if post_grad:
                print(f"       {'':6} Post-power grad: mean={post_grad.get('mean', 0):+.6f}, "
                      f"std={post_grad.get('std', 0):.6f}, max_abs={post_grad.get('max_abs', 0):.6f}")

            # Show post-norm stats if available
            if post_norm:
                norm_mean = post_norm.get('mean', 0)
                norm_std = post_norm.get('std', 0)
                norm_min = post_norm.get('min', 0)
                norm_max = post_norm.get('max', 0)
                print(f"       {'':6} POST-NORM: μ={norm_mean:+.3f} σ={norm_std:.3f} "
                      f"[{norm_min:.3f}, {norm_max:.3f}] ← AFTER LayerNorm")

            print()

    print(f"{'='*80}")

    # Analyze gradient flow health
    analyze_gradient_health(model)


def analyze_gradient_health(model):
    """Analyze whether gradients are vanishing or exploding"""
    print("\nGRADIENT HEALTH CHECK:")
    print("-" * 80)

    grad_norms = []
    for i, transformer_layer in enumerate(model.transformers):
        ff = transformer_layer.ff
        stats = ff.get_gradient_stats()

        pre_grad = stats['gradients'].get('pre_power', {})
        grad_norm = pre_grad.get('norm', 0)
        grad_norms.append(grad_norm)

    if not grad_norms or all(g == 0 for g in grad_norms):
        print("⚠️  WARNING: No gradients recorded (call backward() before logging)")
        return

    # Calculate statistics
    min_grad = min(grad_norms)
    max_grad = max(grad_norms)
    avg_grad = sum(grad_norms) / len(grad_norms)

    # Calculate ratio of first to last layer
    first_layer_grad = grad_norms[0]
    last_layer_grad = grad_norms[-1]
    ratio = first_layer_grad / last_layer_grad if last_layer_grad > 1e-10 else float('inf')

    print(f"Gradient norms: min={min_grad:.6f}, max={max_grad:.6f}, avg={avg_grad:.6f}")
    print(f"First/Last layer ratio: {ratio:.2f}x (first={first_layer_grad:.6f}, last={last_layer_grad:.6f})")

    # Diagnosis
    if ratio > 100:
        print("❌ SEVERE VANISHING GRADIENTS in deep layers!")
        print("   Suggestion: Reduce power progression or increase learning rate for deep layers")
    elif ratio > 10:
        print("⚠️  MODERATE VANISHING GRADIENTS in deep layers")
        print("   Suggestion: Monitor deep layer learning, may need adjustment")
    elif ratio < 0.1:
        print("⚠️  EXPLODING GRADIENTS in deep layers")
        print("   Suggestion: Apply gradient clipping or reduce learning rate")
    else:
        print("✅ Gradients are flowing relatively well across layers")

    # Check for dead layers
    dead_threshold = avg_grad * 0.01 if avg_grad > 0 else 1e-8
    dead_layers = [i+1 for i, g in enumerate(grad_norms) if g < dead_threshold]

    if dead_layers:
        print(f"⚠️  Potentially DEAD layers (very low gradients): {dead_layers}")

    print("-" * 80)


def log_layer_summary(model, step: int):
    """Print a compact summary suitable for frequent logging"""
    grad_norms = []

    for transformer_layer in model.transformers:
        ff = transformer_layer.ff
        stats = ff.get_gradient_stats()
        pre_grad = stats['gradients'].get('pre_power', {})
        grad_norms.append(pre_grad.get('norm', 0))

    if all(g == 0 for g in grad_norms):
        return  # No gradients yet

    # Print compact one-line summary
    grad_str = ' '.join([f"{g:.2e}" for g in grad_norms[::3]])  # Every 3rd layer
    print(f"Step {step:5d} | Grad norms (layers 1,4,7,10): {grad_str}")
