import torch
import torch.nn as nn
import os
import time
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights


def get_model_stats(model_func, weights, name):
    model = model_func(weights=weights)
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6

    path = f"{name}.pt"
    example_input = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, example_input)
    torch.jit.save(traced, path)
    file_size = os.path.getsize(path) / (1024 * 1024)
    if os.path.exists(path):
        os.remove(path)

    start = time.time()
    for _ in range(20):
        with torch.no_grad():
            _ = model(example_input)
    latency = (time.time() - start) / 20 * 1000

    return params, file_size, latency


def run_comparative_analysis():
    print("--- M31: Model Optimization Analysis (R50 vs. Optimized R18) ---")

    params_base, size_base, lat_base = get_model_stats(resnet50, ResNet50_Weights.DEFAULT, "Baseline_R50")

    r18 = resnet18(weights=ResNet18_Weights.DEFAULT)
    optimized_r18 = torch.quantization.quantize_dynamic(r18, {nn.Linear}, dtype=torch.qint8)

    params_opt, size_opt, lat_opt = get_model_stats(lambda weights: optimized_r18, None, "Optimized_R18")

    # Ruff hatasını engellemek için çizgileri 60'a düşürdük
    print("\n" + "=" * 60)
    print(f"{'Metric':<20} | {'Baseline (R50)':<15} | {'Optimized (R18)':<15}")
    print("-" * 60)
    print(f"{'Params (M)':<20} | {params_base:<15.2f} | {params_opt:<15.2f}")
    print(f"{'Size (MB)':<20} | {size_base:<15.2f} | {size_opt:<15.2f}")
    print(f"{'Latency (ms)':<20} | {lat_base:<15.2f} | {lat_opt:<15.2f}")
    reduction = ((size_base - size_opt) / size_base) * 100
    print(f"{'Reduction':<20} | {'0%':<15} | {reduction:<13.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_comparative_analysis()
