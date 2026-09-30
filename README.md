# AniSora on Replicate

Independent AniSora V3.2 image-to-video inference for local Sprute callers.
Unofficial packaging of Bilibili's [Index-AniSora](https://github.com/bilibili/Index-anisora),
using Kijai's FP8 scaled high/low checkpoints and pinned ComfyUI.

## Boundary

The caller supplies an image and prompt. This endpoint returns a preview MP4,
original PNG frame ZIP and effective-settings JSON. Sprute runs background
removal, direction selection and sprite assembly locally.

Defaults match the inference portion of Sprute's turntable workflow: 192x256,
81 frames, UniPC/simple, 8 steps, high/low split at step 3, CFG 1, shift 5,
CLIP vision crop none. Both stages share the seed. High/low CFG and shift can
be adjusted independently. Use `prepared_input=true` for exact opaque input
pixels; otherwise alpha is composited over #808080 and center-cropped to fit.
`reverse_frames=true` reproduces Sprute's reversal before direction extraction.
The default output is chronological model order.

## Build

```
cog build --use-cuda-base-image=false
```

All five weight files in `weights.json` are included during build and checked
at startup. Runtime does not download weights. Worker setup is limited to ten
minutes, prediction to fourteen minutes, and Comfy execution to twelve minutes.
An API `Cancel-After` header is still required to bound platform cold start.

## Status

Public model: https://replicate.com/sprited/anisora

Deployed version:
`2326f1728b7ae815cddaa444c45d3557fb3be6362c02c5c30cfa0a1e031bdee8`.
The published API requires only `image` and returns `video`, `frames`,
`metadata`, and `seed`. Hosted inference verification is running.

Eight tests pass. An offline container HTTP test on gin's RTX PRO 6000 Blackwell
generated 81 PNG frames at 192x256, seed 42, in 18.01 seconds after 6.76 seconds
of setup. No model mounts or network were available. The extracted frames were
visually checked for a full character rotation. This is one local validation,
not a hosted speed estimate. Hosted verification is pending.

## Provenance

AniSora V3 weights/code and the Kijai repack are identified as Apache-2.0 by
their upstream projects. Shared Wan components come from Comfy-Org's Wan 2.1
repack. ComfyUI is GPL-3.0 and runs as a separate subprocess. Packaging utilities
are adapted from [Sprited's SCAIL-2 packaging](https://github.com/sprited-ai/scail-2)
under its MIT license. See `weights.json` for pinned download sources.

## Calling from Sprute

Pin a tested Replicate version when integrating. Send `image`, `seed`,
`return_frames: true` and any sampling overrides. Use `prepared_input: true`
only for an opaque image already at the requested width and height.

The result contains `video`, `frames`, `metadata` and `seed`. Download `frames`
for lossless processing; do not extract sprites from the lossy MP4 preview.
Frame ZIP order is zero-based and follows `reverse_frames`. Inspect metadata
for the effective dimensions, frame count, seed and sampler settings.

For a bounded smoke test, `tools/test_endpoint.py` sends `Cancel-After: 10m`,
polls a single prediction, and cancels it if its local watchdog expires. It
does not automatically retry prediction creation. For example:

```sh
python tools/test_endpoint.py \
  --model sprited/anisora \
  --version TESTED_VERSION_ID \
  --inputs input.json \
  --out output/hosted-test \
  --token-file /path/to/private-token-file
```

`input.json` may use `{"image":{"file":"/absolute/path/reference.png"},"seed":42}`;
the harness uploads that file before submitting the prediction.
