import subprocess
import zipfile
from pathlib import Path

def encode_mp4(frame_glob_pattern: str, n_frames: int, fps: float, out: str, crf: int=15) -> None:
    subprocess.check_call(['ffmpeg', '-nostdin', '-y', '-loglevel', 'error', '-framerate', f'{fps:.6f}', '-start_number', '1', '-i', frame_glob_pattern, '-frames:v', str(n_frames), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(crf), '-preset', 'medium', '-movflags', '+faststart', out], stdin=subprocess.DEVNULL)

def archive_frames(paths, destination):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for i, path in enumerate(paths):
            archive.write(path, f"{i:06d}.png")
