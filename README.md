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

Packaging in development. Pure graph tests pass; container and hosted GPU
validation are pending. Do not infer hosted availability from these files.

## Provenance

AniSora V3 weights/code and the Kijai repack are identified as Apache-2.0 by
their upstream projects. Shared Wan components come from Comfy-Org's Wan 2.1
repack. ComfyUI is GPL-3.0 and runs as a separate subprocess. Packaging utilities
are adapted from [Sprited's SCAIL-2 packaging](https://github.com/sprited-ai/scail-2)
under its MIT license. See `weights.json` for pinned download sources.
