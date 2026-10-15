import torch
from core.serializer import quantize_tensor_int8, dequantize_tensor_int8

tests = {
    "positive only": torch.linspace(1.0, 5.0, 1000),
    "negative only": torch.linspace(-5.0, -1.0, 1000),
    "mixed signs": torch.linspace(-5.0, 5.0, 1000),
    "small positive": torch.linspace(0.001, 0.01, 1000),
    "constant": torch.full((1000,), 3.5),
}

for name, original in tests.items():
    packed = quantize_tensor_int8(original)
    restored = dequantize_tensor_int8(packed)
    error = (original - restored).abs()

    print(f"\n{name}")
    print(f"  Original range: {original.min().item():.6f} to {original.max().item():.6f}")
    print(f"  Scale:          {packed['scale']}")
    print(f"  Zero point:     {packed['zero_point']}")
    print(f"  Max error:      {error.max().item():.6f}")
    print(f"  Mean error:     {error.mean().item():.6f}")
    print(f"  Saturated low:  {(packed['quantized'] == -128).sum().item()}")
    print(f"  Saturated high: {(packed['quantized'] == 127).sum().item()}")
