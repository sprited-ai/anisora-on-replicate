"""AniSora V3.2 high/low-noise sampling, matching Sprute's turntable graph."""
from dataclasses import dataclass

HIGH = 'Wan2_2-I2V_AniSoraV3_2_HIGH_14B_fp8_e4m3fn_scaled_KJ.safetensors'
LOW = 'Wan2_2-I2V_AniSoraV3_2_LOW_14B_fp8_e4m3fn_scaled_KJ.safetensors'
TEXT = 'umt5_xxl_fp8_e4m3fn_scaled.safetensors'
VAE = 'wan_2.1_vae.safetensors'
VISION = 'clip_vision_h.safetensors'

@dataclass
class Params:
    image: str
    prompt: str
    negative_prompt: str
    width: int = 192
    height: int = 256
    num_frames: int = 81
    seed: int = 42
    steps: int = 8
    switch_step: int = 3
    cfg: float = 1
    low_cfg: float = 1
    shift: float = 5
    low_shift: float = 5
    sampler: str = 'uni_pc'
    scheduler: str = 'simple'
    output_prefix: str = 'frames/f'

    def validate(self):
        if not (32 <= self.width <= 1024 and 32 <= self.height <= 1024) or self.width % 16 or self.height % 16:
            raise ValueError('Dimensions must be multiples of 16 between 32 and 1024')
        if not (5 <= self.num_frames <= 161) or (self.num_frames - 1) % 4:
            raise ValueError('num_frames must be 4n+1, between 5 and 161')
        if not 0 < self.switch_step < self.steps <= 100:
            raise ValueError('Require 0 < switch_step < steps <= 100')


def build_graph(p: Params):
    p.validate()
    g = {}
    def node(key, kind, **inputs):
        g[key] = dict(class_type=kind, inputs=inputs)
        return [key, 0]
    high = node('high', 'UNETLoader', unet_name=HIGH, weight_dtype='default')
    low = node('low', 'UNETLoader', unet_name=LOW, weight_dtype='default')
    high = node('high_shift', 'ModelSamplingSD3', model=high, shift=p.shift)
    low = node('low_shift', 'ModelSamplingSD3', model=low, shift=p.low_shift)
    clip = node('clip', 'CLIPLoader', clip_name=TEXT, type='wan', device='default')
    vae = node('vae', 'VAELoader', vae_name=VAE)
    vision = node('vision', 'CLIPVisionLoader', clip_name=VISION)
    image = node('image', 'LoadImage', image=p.image)
    features = node('features', 'CLIPVisionEncode', clip_vision=vision, image=image, crop='none')
    positive = node('positive', 'CLIPTextEncode', text=p.prompt, clip=clip)
    negative = node('negative', 'CLIPTextEncode', text=p.negative_prompt, clip=clip)
    latent = node('conditioning', 'WanImageToVideo', width=p.width, height=p.height,
                  length=p.num_frames, batch_size=1, positive=positive, negative=negative,
                  vae=vae, clip_vision_output=features, start_image=image)
    common = dict(noise_seed=p.seed, steps=p.steps, sampler_name=p.sampler,
                  scheduler=p.scheduler, positive=['conditioning', 0], negative=['conditioning', 1])
    first = node('sample_high', 'KSamplerAdvanced', **common, cfg=p.cfg, model=high,
                 latent_image=['conditioning', 2], add_noise='enable', start_at_step=0,
                 end_at_step=p.switch_step, return_with_leftover_noise='enable')
    second = node('sample_low', 'KSamplerAdvanced', **common, cfg=p.low_cfg, model=low,
                  latent_image=first, add_noise='disable', start_at_step=p.switch_step,
                  end_at_step=p.steps, return_with_leftover_noise='disable')
    frames = node('decode', 'VAEDecode', samples=second, vae=vae)
    node('save', 'SaveImage', images=frames, filename_prefix=p.output_prefix)
    return g
