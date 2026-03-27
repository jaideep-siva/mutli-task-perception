"""Run single-image inference for the minimal Faster R-CNN detection demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.demo_detector import build_demo_detector
from src.utils.config import load_yaml
from src.utils.visualization import draw_pred_boxes


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for checkpointed single-image inference."""

    parser = argparse.ArgumentParser(description="Inference for the detector demo.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. Defaults to the latest checkpoint in the config directory.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    """Selects the runtime device for inference."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Returns the most recent checkpoint file in the configured directory."""

    candidates = sorted(checkpoint_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files were found in '{checkpoint_dir}'.")
    return candidates[-1]


def load_and_resize_image(image_path: Path, image_size: list[int]) -> torch.Tensor:
    """Loads an image, resizes to ``[H, W]``, and converts it to a float tensor in ``[0, 1]``."""

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        resized = image.resize((int(image_size[1]), int(image_size[0])), resample=Image.BILINEAR)
    return F.to_tensor(resized)


def main() -> None:
    """Loads configs and a checkpoint, runs inference, and saves a drawn prediction image."""

    args = parse_args()
    data_config = load_yaml(ROOT / "configs" / "data" / "insta360_detection.yaml")
    train_config = load_yaml(ROOT / "configs" / "train" / "detector_demo.yaml")

    checkpoint_dir = ROOT / train_config["checkpoint_dir"]
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(checkpoint_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    class_map = checkpoint.get("class_map", data_config["class_map"])
    image_size = checkpoint.get("image_size", data_config["image_size"])
    id_to_class = {class_id: name for name, class_id in class_map.items()}

    model = build_demo_detector(num_classes=len(class_map))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_device(str(train_config.get("device", "auto")))
    model.to(device)
    model.eval()

    image_path = Path(args.image)
    image_tensor = load_and_resize_image(image_path, image_size)

    with torch.no_grad():
        prediction = model([image_tensor.to(device)])[0]

    drawn = draw_pred_boxes(
        image_tensor,
        prediction,
        id_to_class=id_to_class,
        score_thresh=float(train_config.get("score_thresh", 0.5)),
    )

    pred_dir = ROOT / train_config["pred_dir"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    output_path = pred_dir / f"{image_path.stem}_prediction.png"
    Image.fromarray(drawn.permute(1, 2, 0).cpu().numpy()).save(output_path)
    print(f"saved prediction: {output_path}")


if __name__ == "__main__":
    main()
