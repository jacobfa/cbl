import os
import time
import statistics
import torch
import torch.nn as nn
from thop import profile

# Import the vanilla Transformer LM from model.py
import model
# Import the CFL-augmented Transformer LM from cfl_model.py
import cfl_model

# ---------------------------------------------------------------------------
# ModelWrapper: Wraps a model so that forward() returns only logits.
# This is necessary for FLOPs counting with THOP.
# ---------------------------------------------------------------------------
class ModelWrapper(nn.Module):
    def __init__(self, model_instance):
        super(ModelWrapper, self).__init__()
        self.model = model_instance

    def forward(self, x):
        logits, _ = self.model(x)
        return logits

# ---------------------------------------------------------------------------
# Measures forward-pass latency over multiple runs and computes average & std.
# ---------------------------------------------------------------------------
def measure_latency(model: nn.Module, inputs, warmup=2, reps=10):
    """
    Run 'warmup' forward passes (not timed), then measure the latency for
    'reps' forward passes (timed individually). Return average & std dev.
    """
    model.eval()
    latencies = []

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(*inputs)

        # Timed runs
        for _ in range(reps):
            start_iter = time.time()
            _ = model(*inputs)
            end_iter = time.time()
            latencies.append(end_iter - start_iter)

    avg_latency = statistics.mean(latencies)
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    return avg_latency, std_latency

# ---------------------------------------------------------------------------
# Computes FLOPs and parameter count using THOP.
# Returns FLOPs in billions (B FLOPs) and parameter count.
# ---------------------------------------------------------------------------
def get_flops_and_params_thop(model: nn.Module, vocab_size=30522, seq_len=32, device='cpu'):
    """
    Wrap the model so only the logits are returned (required by THOP),
    build a dummy input of size (1, seq_len), and measure FLOPs & params.
    """
    wrapped_model = ModelWrapper(model).to(device)

    try:
        flops, params = profile(
            wrapped_model,
            inputs=(torch.randint(0, vocab_size, (1, seq_len), device=device),),
            verbose=False
        )
    except Exception as e:
        print("Flops estimation failed with exception:", e)
        return None, None

    # Convert FLOPs to billions
    flops_in_B = flops / 1e9
    return flops_in_B, params

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # New, larger configurations to "greatly increase" layers & parameter sizes:
    # (embed_dim, num_heads, num_layers, ffn_dim, seq_len)
    configs = [
        (256, 8, 12, 2048, 64),
        (512, 8, 12, 4096, 128),
        (512, 16, 24, 4096, 128),
        (768, 12, 24, 3072, 128),
    ]

    # Set T_refine to 1 and use a smaller context dimension (16)
    T_refine = 1
    context_dim = 16

    log_filename = "log.txt"
    if os.path.exists(log_filename):
        os.remove(log_filename)

    with open(log_filename, "w") as f:
        f.write("Comparison: Vanilla Transformer vs CFL-Transformer\n")
        f.write("-----------------------------------------------------\n")

        for (embed_dim, num_heads, num_layers, ffn_dim, seq_len) in configs:
            f.write(f"\n=== Config: embed_dim={embed_dim}, num_heads={num_heads}, "
                    f"num_layers={num_layers}, ffn_dim={ffn_dim}, seq_len={seq_len} ===\n")

            # 1) Vanilla Transformer from model.py
            vanilla_model = model.SimpleTransformerLM(
                vocab_size=30522,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ffn_dim=ffn_dim,
                dropout=0.1,
                block_size=seq_len
            ).to(device)

            dummy_input_vanilla = torch.randint(0, 30522, (1, seq_len), device=device)
            flops_vanilla, params_vanilla = get_flops_and_params_thop(
                vanilla_model, vocab_size=30522, seq_len=seq_len, device=device
            )
            lat_vanilla, std_vanilla = measure_latency(
                vanilla_model, inputs=(dummy_input_vanilla,), warmup=2, reps=10
            )

            if flops_vanilla is not None and params_vanilla is not None:
                param_million_vanilla = params_vanilla / 1e6
                f.write(
                    f"[Vanilla] Params: {param_million_vanilla:.3f} M, "
                    f"FLOPs: {flops_vanilla:.2f} B, "
                    f"Latency: {lat_vanilla:.6f} s (+/- {std_vanilla:.6f} s)\n"
                )
            else:
                f.write(
                    f"[Vanilla] FLOPs estimation failed, "
                    f"Latency: {lat_vanilla:.6f} s (+/- {std_vanilla:.6f} s)\n"
                )

            # 2) CFL-augmented Transformer from cfl_model.py
            cfl_model_instance = cfl_model.SimpleTransformerLM(
                vocab_size=30522,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ffn_dim=ffn_dim,
                dropout=0.1,
                block_size=seq_len,
                context_dim=context_dim,
                T_refine=T_refine
            ).to(device)

            dummy_input_cfl = torch.randint(0, 30522, (1, seq_len), device=device)
            flops_cfl, params_cfl = get_flops_and_params_thop(
                cfl_model_instance, vocab_size=30522, seq_len=seq_len, device=device
            )
            lat_cfl, std_cfl = measure_latency(
                cfl_model_instance, inputs=(dummy_input_cfl,), warmup=2, reps=10
            )

            if flops_cfl is not None and params_cfl is not None:
                param_million_cfl = params_cfl / 1e6
                f.write(
                    f"[CFL]     Params: {param_million_cfl:.3f} M, "
                    f"FLOPs: {flops_cfl:.2f} B, "
                    f"Latency: {lat_cfl:.6f} s (+/- {std_cfl:.6f} s)\n"
                )
            else:
                f.write(
                    f"[CFL]     FLOPs estimation failed, "
                    f"Latency: {lat_cfl:.6f} s (+/- {std_cfl:.6f} s)\n"
                )

    print(f"Done. Results are in {log_filename}")

if __name__ == "__main__":
    main()
