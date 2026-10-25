"""Find where ETSAM on another device (MPS) departs from the CPU (diagnostics).

    python tests/e2e/compare_devices.py TOMOGRAM.mrc [--device mps] [--slices 3]

Builds ETSAM's stage-1 model on the CPU and on the device, checks that the
loaded weights match, runs a few slices on both with a forward hook on every
module, and prints the first module calls whose outputs differ. Also checks
single operations ETSAM relies on (complex rotary encoding, bfloat16 copies,
scaled dot-product attention).
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mrcfile  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import etsam  # noqa: E402
import inference  # noqa: E402
from sam2.utils.volume import NormalizedTomogram  # noqa: E402


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)


def summary(value):
    out = []
    for t in tensors(value):
        t = t.detach().to("cpu").double()
        if t.is_complex():
            t = torch.view_as_real(t)
        out.append((tuple(t.shape), t.mean().item(), t.std().item() if t.numel() > 1 else 0.0,
                    t.abs().max().item() if t.numel() else 0.0))
    return out


def check_ops(device):
    print("== single operations ==")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 8, 64, 32, generator=g)
    angles = torch.randn(64, 16, generator=g)

    def rotary(x, angles):
        freqs = torch.polar(torch.ones_like(angles), angles)
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(xc * freqs).flatten(3)

    def report(name, cpu, dev):
        dev = dev.to("cpu").float()
        cpu = cpu.to(torch.float32)
        print(f"{name}: max abs difference {(cpu - dev).abs().max().item():.3g}")

    report("polar", torch.polar(torch.ones_like(angles), angles).real,
           torch.polar(torch.ones_like(angles).to(device), angles.to(device)).real)
    report("rotary encoding", rotary(x, angles), rotary(x.to(device), angles.to(device)))
    report("bfloat16 to device", x.bfloat16(), x.bfloat16().to(device))
    report("bfloat16 on device", x.bfloat16(), x.to(device).bfloat16())
    big = torch.randn(1000, generator=g)
    report("copy of a view with a storage offset", big[100:356], big[100:356].to(device))
    q, k, v = (torch.randn(1, 8, 300, 64, generator=g) for _ in range(3))
    report("scaled dot-product attention",
           torch.nn.functional.scaled_dot_product_attention(q, k, v),
           torch.nn.functional.scaled_dot_product_attention(q.to(device), k.to(device), v.to(device)))
    q, k, v = (torch.randn(1, 1, 4096, 256, generator=g) for _ in range(3))
    report("scaled dot-product attention (4096 tokens)",
           torch.nn.functional.scaled_dot_product_attention(q, k, v),
           torch.nn.functional.scaled_dot_product_attention(q.to(device), k.to(device), v.to(device)))


def trace(device, volume, ckpt):
    records, models = [], []
    build = inference.build_sam2_video_predictor

    def traced_build(*args, **kwargs):
        model = build(*args, **kwargs)
        models.append(model)
        for name, module in model.named_modules():
            module.register_forward_hook(
                lambda mod, inputs, output, name=name: records.append((name, summary(output)))
            )
        return model

    inference.build_sam2_video_predictor = traced_build
    try:
        output = np.empty(volume.shape, dtype=np.float32)
        inference.etsam_inference(
            volume, config_path=etsam.CONFIG, ckpt_path=ckpt, prompt_method="grid_zero",
            grid_interval=50, device=device, output=output,
        )
    finally:
        inference.build_sam2_video_predictor = build
    return records, output, {k: v.detach().to("cpu") for k, v in models[0].state_dict().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tomogram")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--slices", type=int, default=3)
    parser.add_argument("--show", type=int, default=40)
    args = parser.parse_args()
    print("PyTorch", torch.__version__)
    check_ops(args.device)

    data = mrcfile.read(args.tomogram)
    data = data[len(data) - args.slices:]
    volume = NormalizedTomogram(np.ascontiguousarray(data))
    ckpt, _ = etsam.ensure_checkpoints()
    cpu_records, cpu_output, cpu_state = trace("cpu", volume, ckpt)
    dev_records, dev_output, dev_state = trace(args.device, volume, ckpt)

    print("== weights after loading ==")
    bad = [(k, (cpu_state[k].float() - dev_state[k].float()).abs().max().item())
           for k in cpu_state if not torch.equal(cpu_state[k], dev_state[k])]
    print(f"{len(bad)} of {len(cpu_state)} tensors differ")
    for k, d in bad[:20]:
        print(f"  {k}: max abs difference {d:.3g}")

    print("== module outputs ==")
    print(f"{len(cpu_records)} calls on the CPU, {len(dev_records)} on {args.device}")
    shown = 0
    for index, ((name, a), (name2, b)) in enumerate(zip(cpu_records, dev_records)):
        if name != name2 or len(a) != len(b):
            print(f"call {index}: different calls: {name} / {name2}")
            break
        for (shape, mean, std, top), (shape2, mean2, std2, top2) in zip(a, b):
            scale = max(std, abs(mean), 1e-6)
            if shape != shape2 or abs(mean - mean2) / scale > 0.05 or abs(std - std2) / scale > 0.05:
                print(f"call {index} {name or '(model)'} {shape}: cpu mean {mean:.4g} std {std:.4g} "
                      f"max {top:.4g} | {args.device} mean {mean2:.4g} std {std2:.4g} max {top2:.4g}")
                shown += 1
                break
        if shown >= args.show:
            break
    if shown == 0:
        print("no module output differs by more than 5%")
    print("== logits ==")
    for index in range(len(cpu_output)):
        print(f"slice {index}: cpu mean {cpu_output[index].mean():.4g} | {args.device} mean "
              f"{dev_output[index].mean():.4g} | max abs difference "
              f"{np.abs(cpu_output[index] - dev_output[index]).max():.4g}")


if __name__ == "__main__":
    main()
