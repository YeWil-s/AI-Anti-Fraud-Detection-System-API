from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import build_model


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model = build_model(
        backbone_name=config.get("backbone_name", "xception"),
        pretrained=False,
        hidden_dim=int(config.get("hidden_dim", 256)),
        lstm_layers=int(config.get("lstm_layers", 1)),
        dropout=float(config.get("dropout", 0.3)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_ckpt = root / "training_module" / "outputs" / "xception_lstm" / "best.pth"
    default_onnx = root / "models" / "deepfake_xception_lstm.onnx"

    parser = argparse.ArgumentParser(description="Export trained deepfake video model to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=default_ckpt)
    parser.add_argument("--output", type=Path, default=default_onnx)
    parser.add_argument("--img-size", type=int, default=0, help="Override checkpoint img_size when > 0.")
    parser.add_argument("--seq-len", type=int, default=0, help="Override checkpoint seq_len when > 0.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dynamic-batch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()

    model, config = load_checkpoint_model(checkpoint_path, device)
    img_size = args.img_size if args.img_size > 0 else int(config.get("img_size", 299))
    seq_len = args.seq_len if args.seq_len > 0 else int(config.get("seq_len", 10))

    dummy_input = torch.randn(1, seq_len, 3, img_size, img_size, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "input_sequence": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input_sequence"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output ONNX: {output_path}")
    print(f"Input shape: (1, {seq_len}, 3, {img_size}, {img_size})")
    print("Logits order: [FAKE, REAL]")


if __name__ == "__main__":
    main()
