# AniSora V3.2

Animate an image with a text prompt. Upload a reference image and describe the motion, scene, and camera behavior you want.

Unofficial community deployment of [Bilibili's Index-AniSora](https://github.com/bilibili/Index-anisora).

## Examples

[View the examples](https://replicate.com/sprited/anisora/examples), including a photoreal fox and an illustrated explorer. Examples use MP4 output without a PNG frame archive.

## Quantization

This deployment uses **Kijai's FP8-scaled quantized high-noise and low-noise model weights** to reduce GPU memory usage. Outputs may differ from the original full-precision weights. FP8 refers to the model weights; it does not mean every operation or component uses FP8.

## Inputs and outputs

The image is required. Supply a prompt describing the desired video; the default prompt produces a character turntable. Width, height, frame count, seed, sampling steps, and the high/low stage transition can be adjusted.

The output includes an MP4, WebM, or animated WebP, a ZIP of original PNG frames when return_frames is enabled, effective settings JSON, and the seed. Use the PNG frames for editing that needs to avoid video compression artifacts.

81 frames at 24 fps produces approximately 3.4 seconds of video. The fps setting controls playback rate; it does not change the generated motion.

Cold starts can take several minutes. API callers can use the per-request `Cancel-After: 10m` header to bound startup and inference. This is not an automatic Playground timeout.

[Deployment source and component provenance](https://github.com/sprited-ai/anisora-on-replicate).

## Output format and quality

Choose `output_format`: `mp4` (default), `webm`, or animated `webp`. `output_quality` sets compression quality from 1 to 100 (default 80). Higher values generally produce larger files; values are not comparable across codecs and do not measure inference quality. For a video without additional frame archives, set `return_frames` to false. WebP output does not automatically remove the background.
