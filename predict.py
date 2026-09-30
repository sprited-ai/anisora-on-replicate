"""Independent AniSora GPU inference. Sprite processing belongs to the caller."""
import json
import os
from pathlib import Path as FilePath
import secrets
import shutil
import time
import uuid
from typing import Optional

from cog import BasePredictor, BaseModel, Input, Path
from PIL import Image, ImageOps
from comfy_client import ComfyServer
from runtime_limits import deadline
from output_media import encode_animation
from workflow import Params, build_graph
from media import archive_frames

ROOT = FilePath(os.environ.get('ANISORA_WORK', '/tmp/anisora'))
COMFY = FilePath(os.environ.get('COMFY_DIR', '/ComfyUI'))

class Output(BaseModel):
    video: Path
    metadata: Path
    seed: int
    frames: Optional[Path] = None

class Predictor(BasePredictor):
    def setup(self):
        with deadline(600, 'setup'):
            manifest = json.loads(FilePath('/src/weights.json').read_text())
            for record in manifest:
                path = COMFY / 'models' / record['path']
                if not path.is_file() or path.stat().st_size != record['size']:
                    raise RuntimeError(f'Missing or truncated bundled weight: {path}')
            self.comfy = ComfyServer(str(COMFY), str(ROOT/'input'), str(ROOT/'output'),
                                     str(ROOT/'temp'), extra_args=tuple(os.environ.get('COMFY_EXTRA_ARGS', '').split()))
            self.comfy.start()
            self.comfy.wait_ready()
            print('[setup] AniSora ready; all weights bundled', flush=True)

    def predict(
        self,
        image: Path = Input(description='Reference image. Alpha is composited over gray; preprocessed images can be passed unchanged.'),
        prompt: str = Input(default="A 360-degree turning and circling video of a 2D RPG game character. Pixel Art. Game Sprite. This is a full-body long shot with a plain gray background. Preserve character's reference proportions, clothes and hairstyle. Clear readable silhouette, clean contours, flat local colors with restrained cel shading. Fixed camera, constant character size, feet on the same spot. aesthetic score: 5.5. motion score: 3.0. There is no text in the video.", description='Describe the video. Default is Sprute character turntable.'),
        negative_prompt: str = Input(default='3D, maya, render, blender, photorealistic, glossy plastic, sculpted figurine, dramatic lighting, gradients, walking, running, dancing, camera movement, zoom, cropped feet, outfit change, extra limbs, text, scene change, dithering, '),
        width: int = Input(default=192, ge=32, le=1024),
        height: int = Input(default=256, ge=32, le=1024),
        num_frames: int = Input(default=81, ge=5, le=161, description='Must be 4n+1.'),
        fps: int = Input(default=24, ge=1, le=60, description='Playback FPS; does not change sampled motion.'),
        steps: int = Input(default=8, ge=2, le=100),
        switch_step: int = Input(default=3, ge=1, le=99, description='Step where high-noise model hands off to low-noise model.'),
        guidance_scale: float = Input(default=1, ge=0, le=20),
        low_guidance_scale: float = Input(default=1, ge=0, le=20),
        shift: float = Input(default=5, ge=0.01, le=20),
        low_shift: float = Input(default=5, ge=0.01, le=20),
        sampler_name: str = Input(default='uni_pc', choices=['uni_pc', 'euler']),
        scheduler: str = Input(default='simple', choices=['simple', 'normal', 'karras', 'exponential', 'sgm_uniform', 'ddim_uniform', 'beta', 'linear_quadratic', 'kl_optimal']),
        prepared_input: bool = Input(default=False, description='Preserve exact opaque RGB pixels. Image must match width/height.'),
        reverse_frames: bool = Input(default=False, description='Reverse output ordering. Sprute currently reverses AniSora output before direction extraction.'),
        output_format: str = Input(default="mp4", choices=["mp4", "webm", "webp"], description="Output file format: MP4 (H.264), WebM (VP9), or looping animated WebP. Does not change inference."),
        output_quality: int = Input(default=80, ge=1, le=100, description="Compression quality: higher means better fidelity and usually larger files. Codec-relative, not a model quality score; 100 does not guarantee lossless output. Use PNG frames for lossless originals."),
        return_frames: bool = Input(default=True, description='Return lossless PNG ZIP for local background removal and sprite assembly.'),
        seed: Optional[int] = Input(default=None),
    ) -> Output:
        with deadline(840, 'predict'):
            started = time.monotonic()
            seed = secrets.randbits(32) if seed is None else seed
            job = uuid.uuid4().hex
            p = Params(image=f'{job}.png', prompt=prompt, negative_prompt=negative_prompt,
                       width=width, height=height, num_frames=num_frames, seed=seed,
                       steps=steps, switch_step=switch_step, cfg=guidance_scale,
                       low_cfg=low_guidance_scale, shift=shift, low_shift=low_shift,
                       sampler=sampler_name, scheduler=scheduler, output_prefix=f'{job}/f')
            graph = build_graph(p)
            if not self.comfy.alive():
                self.comfy.start()
                self.comfy.wait_ready()
            source = ROOT/'input'/p.image
            output = ROOT/'output'/job
            result = ROOT/'results'/job
            result.mkdir(parents=True)
            try:
                with Image.open(str(image)) as im:
                    rgba = im.convert('RGBA')
                    if prepared_input:
                        if im.size != (width, height) or rgba.getchannel('A').getextrema()[0] != 255:
                            raise ValueError('Prepared image must be opaque and match width/height')
                        rgb = im.convert('RGB')
                    else:
                        bg = Image.new('RGBA', im.size, (128,128,128,255))
                        rgb = ImageOps.fit(Image.alpha_composite(bg, rgba).convert('RGB'), (width,height), method=Image.Resampling.LANCZOS)
                    rgb.save(source)
                print(f'[sample] seed={seed} {width}x{height} frames={num_frames} steps={steps} split={switch_step}', flush=True)
                self.comfy.run(graph, timeout=720)
                paths = sorted(output.glob('f_*_.png'))
                if len(paths) != num_frames:
                    raise RuntimeError(f'Expected {num_frames} frames, received {len(paths)}')
                if reverse_frames:
                    paths.reverse()
                ordered = result/'png'; ordered.mkdir()
                for i, path in enumerate(paths):
                    shutil.copyfile(path, ordered/f'f_{i+1:05d}_.png')
                video = result/f'video.{output_format}'
                encode_animation(str(ordered/'f_%05d_.png'), num_frames, fps, str(video), output_format, output_quality)
                archive = result/'frames.zip' if return_frames else None
                if archive:
                    archive_frames(paths, archive)
                metadata = result/'metadata.json'
                metadata.write_text(json.dumps(dict(model='AniSora V3.2 FP8 scaled', **vars(p), fps=fps,
                                                     output_format=output_format, output_quality=output_quality,
                                                     reverse_frames=reverse_frames, elapsed_seconds=time.monotonic()-started), indent=2))
                return Output(video=Path(video), frames=Path(archive) if archive else None,
                              metadata=Path(metadata), seed=seed)
            finally:
                source.unlink(missing_ok=True)
                shutil.rmtree(output, ignore_errors=True)
                shutil.rmtree(result/'png', ignore_errors=True)
