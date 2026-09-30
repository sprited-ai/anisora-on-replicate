import ast
import contextlib
import json
import secrets
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from PIL import Image, ImageOps
import pytest
from defaults import PROMPT, NEGATIVE_PROMPT
from workflow import Params, build_graph
from media import archive_frames

@pytest.mark.parametrize('reverse', [False, True])
def test_real_prediction_returns_original_ordered_pngs(tmp_path, reverse):
    (tmp_path/'input').mkdir();(tmp_path/'output').mkdir()
    source=tmp_path/'reference.png';Image.new('RGB',(192,256),(10,20,30)).save(source)
    class Output:
        def __init__(self,**values):self.__dict__.update(values)
    class Comfy:
        def alive(self):
            return True
        def run(self, graph, timeout):
            assert timeout==720
            assert graph['sample_high']['inputs']['end_at_step']==3
            dest=tmp_path/'output'/graph['save']['inputs']['filename_prefix']
            dest.parent.mkdir()
            for i in range(5):Image.new('RGB',(192,256),(i*40,20,30)).save(str(dest)+f'_{i+1:05d}_.png')
    tree=ast.parse(Path('predict.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Predictor')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='predict')
    ns=dict(deadline=lambda *a:contextlib.nullcontext(),ROOT=tmp_path,Path=Path,Input=lambda default=None,**kw:default,
            Optional=Optional,Output=Output,Params=Params,build_graph=build_graph,Image=Image,ImageOps=ImageOps,
            time=time,secrets=secrets,uuid=uuid,shutil=shutil,json=json,PROMPT=PROMPT,NEGATIVE_PROMPT=NEGATIVE_PROMPT,
            archive_frames=archive_frames,encode_mp4=lambda pattern,n,fps,path:Path(path).write_bytes(b'preview'))
    exec(compile(ast.Module(body=[method],type_ignores=[]),'predict.py','exec'),ns)
    result=ns['predict'](SimpleNamespace(comfy=Comfy()),image=source,num_frames=5,seed=42,reverse_frames=reverse,prepared_input=True)
    assert result.video.is_file()
    meta=json.loads(result.metadata.read_text());assert meta['seed']==42 and meta['reverse_frames']==reverse
    with zipfile.ZipFile(result.frames) as z:
        from io import BytesIO
        assert len(z.namelist())==5
        first=Image.open(BytesIO(z.read('000000.png')))
        assert first.getpixel((0,0))[0]==(160 if reverse else 0)
    assert not list((tmp_path/'input').iterdir()) and not list((tmp_path/'output').iterdir())
