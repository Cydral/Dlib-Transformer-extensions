#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Reference vision encoding, for comparison with the C++ tower.
#
# The C++ side reports statistics of its visual embeddings and the first values of the
# first token. This script reports the same figures from the reference implementation on
# the same image, so a disagreement points at a stage rather than at a feeling. It is the
# counterpart of slm_reference_chat.py for the half of a multimodal model that has no
# tokens to compare.
#
# Two things it settles that no amount of reading settles.
#
# The scale. A projector output has to arrive in the embedding space of the decoder that
# will receive it, and there is no way to tell from the container whether a root mean
# square of seven is what that space expects or two orders of magnitude too much.
#
# The channel order of the pixel shuffle. Folding a 4 by 4 neighbourhood into the channels
# can interleave the spatial offsets or group them, and both orderings produce output with
# the same statistics; only the trained projector knows which one it was fitted against.
# A mistake here yields a plausible description of the wrong picture.
#
# Preprocessing is reported separately and can be taken from either side, because a
# mismatch there would otherwise be blamed on the tower. --preprocess simple applies the
# same resize and normalization the C++ path applies; --preprocess processor uses the one
# the model ships, which also decides whether an image is split into crops.
#
# Requirements:
#   pip install "torch>=2.2" "transformers>=4.46" "pillow>=10" numpy
#   Only for --preprocess processor: pip install torchvision
#
#   torchvision is what the model's own image processor is built on, and it is not needed
#   for the comparison that matters: --preprocess simple feeds both sides the same tensor
#   and takes its two constants from the command line, the container having declared them.
#
# Usage:
#   slm_reference_vision.py --model HuggingFaceTB/SmolVLM-256M-Instruct --image photo.bmp
#   slm_reference_vision.py --model ... --image photo.bmp --preprocess processor

import argparse
import sys

try:
    import numpy as np
    import torch
    import transformers
    from PIL import Image
except ImportError as exc:
    sys.exit(f"missing dependency: {exc}\n"
             'install with: pip install "torch>=2.2" "transformers>=4.46" '
             '"pillow>=10" "jinja2>=3.1" numpy')

# The auto class for this family was renamed: AutoModelForVision2Seq became
# AutoModelForImageTextToText. Looked up rather than imported, so the script works either
# side of the rename instead of failing at import with a message about a dependency it
# actually has.
_AUTO_CLASS = None
for _name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
    _AUTO_CLASS = getattr(transformers, _name, None)
    if _AUTO_CLASS is not None:
        break
if _AUTO_CLASS is None:
    sys.exit(f"transformers {transformers.__version__} exposes neither "
             "AutoModelForImageTextToText nor AutoModelForVision2Seq")


def report(name, t):
    """Range, mean and root mean square, in the layout the C++ side prints."""
    a = t.detach().to(torch.float32).cpu().numpy().ravel()
    finite = np.isfinite(a)
    bad = int((~finite).sum())
    a = a[finite]
    print(f"{name:<19}: {tuple(t.shape)}  min {a.min():.6g}, max {a.max():.6g}, "
          f"mean {a.mean():.6g}, rms {np.sqrt((a * a).mean()):.6g}"
          + (f"  {bad} NON-FINITE VALUES" if bad else ""))


def simple_preprocess(image, side, mean, std):
    """The resize and normalization the C++ path applies, and nothing else.

    Bilinear, no crop splitting, no aspect preservation. Feeding both sides the same
    tensor is what separates a preprocessing difference from a tower difference.
    """
    img = image.convert("RGB").resize((side, side), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0          # H, W, C
    a = (a - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    a = np.transpose(a, (2, 0, 1))                          # C, H, W
    return torch.from_numpy(a).unsqueeze(0)                 # 1, C, H, W


def main():
    p = argparse.ArgumentParser(
        description="Reference vision encoding, for comparison with the C++ tower.")
    p.add_argument("--model", required=True, help="model id or local path")
    p.add_argument("--image", required=True)
    p.add_argument("--preprocess", default="simple", choices=["simple", "processor"],
                   help="whose image preparation to use (default: simple, the one the "
                        "C++ path applies)")
    p.add_argument("--mean", default="0.5,0.5,0.5",
                   help="per-channel normalization mean for --preprocess simple; the "
                        "container declares it and the C++ path reads it from there")
    p.add_argument("--std", default="0.5,0.5,0.5",
                   help="per-channel normalization deviation for --preprocess simple")
    p.add_argument("--tokens", type=int, default=8,
                   help="values of the first visual token to print (default: 8)")
    p.add_argument("--show-prompt", action="store_true",
                   help="render the chat template with one image and print it, which is "
                        "what settles how many placeholders an image expands into")
    args = p.parse_args()

    if args.show_prompt:
        # The expansion of an image placeholder is a property of the processor, not of the
        # container: how many stand-in tokens one image becomes, and what wraps them, is
        # decided here and nowhere else. Printing it is what lets the C++ template be
        # checked against the thing it imitates.
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(args.model)
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": "What is in this image?"}]}]
        rendered = proc.apply_chat_template(messages, add_generation_prompt=True)
        print("Template rendering, before any image is supplied:")
        print(" ", repr(rendered))
        print(f"  <image> occurrences: {rendered.count('<image>')}")

        # The expansion happens here and not above: apply_chat_template lays out the
        # conversation, the processor replaces each placeholder by as many stand-in tokens
        # as the tower will produce vectors. That count and its wrapping are what a caller
        # has to reserve, so they are read from the thing that decides them.
        try:
            image = Image.open(args.image).convert("RGB")
        except Exception as exc:
            print(f"\n  pass --image to see the expansion ({exc})")
            return
        batch = proc(text=[rendered], images=[[image]], return_tensors="pt")
        ids = batch["input_ids"][0].tolist()
        tok = getattr(proc, "tokenizer", None)
        print(f"\nAfter the processor, with one image: {len(ids)} tokens")
        if tok is not None:
            expanded = tok.decode(ids)
            print(" ", repr(expanded[:160]))
            print("  ...")
            print(" ", repr(expanded[-90:]))
            for mark in ("<image>", "<fake_token_around_image>", "<global-img>",
                         "<row_1_col_1>"):
                n = expanded.count(mark)
                if n:
                    print(f"  {mark:28s} x {n}")
        return

    print(f"Loading {args.model} ...")
    print(f"  transformers {transformers.__version__}, "
          f"loading through {_AUTO_CLASS.__name__}")
    try:
        model = _AUTO_CLASS.from_pretrained(args.model, dtype=torch.float32)
    except TypeError:
        # torch_dtype before it became dtype.
        model = _AUTO_CLASS.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    # The vision half sits under different attribute names across versions; look for it
    # rather than assume, so a rename does not read as a missing model.
    root = getattr(model, "model", model)
    vision = getattr(root, "vision_model", None)
    connector = getattr(root, "connector", None)
    if vision is None or connector is None:
        sys.exit("this model does not expose a vision_model and a connector; "
                 f"available attributes: {[a for a in dir(root) if not a.startswith('_')][:40]}")

    cfg = vision.config
    side = getattr(cfg, "image_size", 512)
    print(f"Vision tower       : image {side}, patch {getattr(cfg, 'patch_size', '?')}, "
          f"width {getattr(cfg, 'hidden_size', '?')}, "
          f"{getattr(cfg, 'num_hidden_layers', '?')} layers")

    image = Image.open(args.image)
    print(f"Image              : {args.image} ({image.width}x{image.height})")

    mean = [float(v) for v in args.mean.split(",")]
    std = [float(v) for v in args.std.split(",")]

    if args.preprocess == "processor":
        # Loaded only on request. Its image processor pulls in torchvision, which this
        # comparison does not need: the mode that matters feeds both sides the same tensor,
        # and the two constants it takes from the processor are declared by the container
        # as well.
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(args.model)
        except Exception as exc:
            sys.exit(f"the model's own preprocessing is unavailable: {exc}\n"
                     "install it with: pip install torchvision\n"
                     "or stay on --preprocess simple, which needs nothing and is what the "
                     "C++ path applies")
        ip = getattr(processor, "image_processor", processor)
        mean = list(getattr(ip, "image_mean", mean))
        std = list(getattr(ip, "image_std", std))
        print(f"Preprocessing      : the model's own (mean {mean}, std {std})")
        out = ip(images=[image], return_tensors="pt", do_image_splitting=False)
        pixel_values = out["pixel_values"]
        # Idefics3 carries a per-image crop axis; a single view collapses it.
        while pixel_values.dim() > 4:
            pixel_values = pixel_values.squeeze(1)
    else:
        print(f"Preprocessing      : simple resize and normalize (mean {mean}, std {std})")
        pixel_values = simple_preprocess(image, side, mean, std)

    report("prepared image", pixel_values)

    with torch.no_grad():
        features = vision(pixel_values=pixel_values).last_hidden_state
        report("tower output", features)
        visual = connector(features)

    if visual.dim() == 3:
        visual = visual.squeeze(0)
    report("visual embeddings", visual)

    n = min(args.tokens, visual.shape[-1])
    print(f"\nFirst visual token, first {n} of {visual.shape[-1]} values:")
    print("  " + " ".join(f"{v:.6g}" for v in visual[0, :n].tolist()))

    # The number the C++ side cannot judge on its own: what the decoder's own token
    # embeddings look like. A projector output two orders of magnitude away from them
    # would drown the text it is spliced into.
    emb = None
    for attr in ("embed_tokens", "get_input_embeddings"):
        obj = getattr(root, attr, None) or getattr(model, attr, None)
        if obj is not None:
            emb = obj() if callable(obj) and attr == "get_input_embeddings" else obj
            break
    if emb is not None and hasattr(emb, "weight"):
        report("token embeddings", emb.weight)
        print("\nThe two lines above are the comparison that matters: visual embeddings "
              "are\nspliced into the same sequence as token embeddings, so their scales "
              "have to\nbe of the same order.")


if __name__ == "__main__":
    main()
