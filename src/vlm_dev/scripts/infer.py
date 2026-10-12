from __future__ import annotations

import argparse

from vlm_dev.utils.config import load_config
from vlm_dev.utils.inference import VLMInference


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a response for one image with a VLM.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML configuration.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional directory containing checkpoint.pt; omit to test the random projector.",
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--prompt", default="Describe this image.", help="Instruction for the model.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, or cpu.")
    args = parser.parse_args()

    engine = VLMInference(load_config(args.config), args.checkpoint, args.device)
    response = engine.generate(
        image_path=args.image,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )
    print(response)


if __name__ == "__main__":
    main()
