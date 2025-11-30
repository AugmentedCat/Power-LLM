"""
Lightweight weight and activation tracking during training.

Logs to a CSV file for easy analysis and plotting.
"""

import torch
import csv
import os
from datetime import datetime


def log_weight_and_activation_stats(model, step, output_file='weight_activation_log.csv'):
    """
    Log weight norms and estimated activation magnitudes.

    This is called periodically (e.g., every 500 steps) to track:
    1. Weight magnitudes per layer
    2. Maximum expected activation (based on weight norms)
    3. Weight growth rate

    Args:
        model: The transformer model
        step: Current training step
        output_file: CSV file to append stats to
    """

    # Check if file exists to write header
    file_exists = os.path.exists(output_file)

    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if new file
        if not file_exists:
            header = ['timestamp', 'step', 'layer', 'power']
            header += ['w1_norm', 'w2_norm', 'w1_max', 'w2_max']
            header += ['w1_mean', 'w1_std', 'w2_mean', 'w2_std']
            writer.writerow(header)

        # Get timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Log each layer
        for i, transformer in enumerate(model.transformers):
            layer_num = i + 1
            power = transformer.ff.activation.power

            # Linear1 weights
            w1 = transformer.ff.linear1.weight.data
            w1_norm = w1.norm().item()
            w1_max = w1.abs().max().item()
            w1_mean = w1.mean().item()
            w1_std = w1.std().item()

            # Linear2 weights
            w2 = transformer.ff.linear2.weight.data
            w2_norm = w2.norm().item()
            w2_max = w2.abs().max().item()
            w2_mean = w2.mean().item()
            w2_std = w2.std().item()

            # Write row
            row = [timestamp, step, layer_num, power]
            row += [w1_norm, w2_norm, w1_max, w2_max]
            row += [w1_mean, w1_std, w2_mean, w2_std]
            writer.writerow(row)


def log_activation_samples(model, input_tensor, step, output_file='activation_log.csv'):
    """
    Log actual activation magnitudes during a forward pass.

    This samples activations from a real batch to track actual values.
    Only called periodically to avoid overhead.

    Args:
        model: The transformer model
        input_tensor: Sample input batch
        step: Current training step
        output_file: CSV file to append stats to
    """

    # Check if file exists
    file_exists = os.path.exists(output_file)

    # Do forward pass to capture activations
    model.eval()
    with torch.no_grad():
        # Get embeddings
        x = model.token_embedding(input_tensor) + model.positional_embedding(input_tensor, 0)
        mask = model.pad_masking(input_tensor, 0)
        if not model.bidirectional:
            mask = mask + model.future_masking(input_tensor, 0)

        activation_stats = []

        # Track through each layer
        for i, transformer in enumerate(model.transformers):
            layer_num = i + 1
            power = transformer.ff.activation.power

            # Attention
            a = transformer.ln_attn(x)
            a, _ = transformer.attn(a, a, a, None, mask)
            x = x + a

            # Feedforward - track activations
            ff_input = transformer.ln_ff(x)
            linear1_out = transformer.ff.linear1(ff_input)

            # Power transformation
            x_powered = torch.pow(linear1_out, power)

            # ReLU
            relu_out = transformer.ff.activation.relu(x_powered)

            # Linear2
            linear2_out = transformer.ff.linear2(relu_out)

            # Residual
            x = x + linear2_out

            # Collect stats
            stats = {
                'layer': layer_num,
                'power': power,
                'pre_power_max': linear1_out.abs().max().item(),
                'pre_power_mean': linear1_out.mean().item(),
                'pre_power_std': linear1_out.std().item(),
                'post_power_max': x_powered.abs().max().item(),
                'post_power_mean': x_powered.mean().item(),
                'post_power_std': x_powered.std().item(),
                'post_relu_max': relu_out.abs().max().item(),
                'post_relu_sparsity': (relu_out == 0).float().mean().item() * 100,
            }
            activation_stats.append(stats)

    model.train()

    # Write to CSV
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if new file
        if not file_exists:
            header = ['timestamp', 'step', 'layer', 'power']
            header += ['pre_power_max', 'pre_power_mean', 'pre_power_std']
            header += ['post_power_max', 'post_power_mean', 'post_power_std']
            header += ['post_relu_max', 'post_relu_sparsity']
            writer.writerow(header)

        # Get timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write rows
        for stats in activation_stats:
            row = [timestamp, step, stats['layer'], stats['power']]
            row += [stats['pre_power_max'], stats['pre_power_mean'], stats['pre_power_std']]
            row += [stats['post_power_max'], stats['post_power_mean'], stats['post_power_std']]
            row += [stats['post_relu_max'], stats['post_relu_sparsity']]
            writer.writerow(row)


def print_tracking_summary(weight_log='weight_activation_log.csv', activation_log='activation_log.csv'):
    """
    Print a summary of tracked statistics for quick monitoring.
    """
    import pandas as pd

    print("\n" + "="*80)
    print("WEIGHT AND ACTIVATION TRACKING SUMMARY")
    print("="*80)

    # Read weight log
    if os.path.exists(weight_log):
        df_weights = pd.read_csv(weight_log)

        if len(df_weights) > 0:
            # Get latest and earliest for each layer
            latest_step = df_weights['step'].max()
            earliest_step = df_weights['step'].min()

            print(f"\nWeight Growth (Step {earliest_step} → {latest_step}):")
            print(f"{'Layer':<8} {'Power':<7} {'Initial W1':<12} {'Current W1':<12} {'Growth':<10}")
            print("-"*60)

            for layer in range(1, 13):
                layer_data = df_weights[df_weights['layer'] == layer]
                if len(layer_data) > 0:
                    earliest = layer_data[layer_data['step'] == earliest_step].iloc[0]
                    latest = layer_data[layer_data['step'] == latest_step].iloc[0]

                    growth = latest['w1_norm'] / earliest['w1_norm'] if earliest['w1_norm'] > 0 else 1.0

                    print(f"{layer:<8} {int(latest['power']):<7} {earliest['w1_norm']:<12.4f} "
                          f"{latest['w1_norm']:<12.4f} {growth:<10.3f}x")

    # Read activation log
    if os.path.exists(activation_log):
        df_act = pd.read_csv(activation_log)

        if len(df_act) > 0:
            latest_step = df_act['step'].max()
            latest_act = df_act[df_act['step'] == latest_step]

            print(f"\nCurrent Activations (Step {latest_step}):")
            print(f"{'Layer':<8} {'Power':<7} {'Pre-Power Max':<15} {'Post-Power Max':<15} {'Sparsity':<10}")
            print("-"*70)

            for layer in range(1, 13):
                layer_act = latest_act[latest_act['layer'] == layer]
                if len(layer_act) > 0:
                    row = layer_act.iloc[0]
                    print(f"{layer:<8} {int(row['power']):<7} {row['pre_power_max']:<15.2e} "
                          f"{row['post_power_max']:<15.2e} {row['post_relu_sparsity']:<10.1f}%")

            # Check for overflow risk
            max_activation = latest_act['post_power_max'].max()
            overflow_limit = 1.8e19  # Squaring limit
            margin = overflow_limit / max_activation

            print()
            print(f"Maximum activation: {max_activation:.2e}")
            print(f"Overflow limit:     {overflow_limit:.2e}")
            print(f"Safety margin:      {margin:.2e}x")

            if margin < 100:
                print("⚠️  WARNING: Close to overflow!")
            elif margin < 10000:
                print("⚠️  Monitor closely")
            else:
                print("✅ Safe")

    print("="*80 + "\n")
