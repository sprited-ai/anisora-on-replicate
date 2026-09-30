import json
from pathlib import Path
import pytest
from workflow import Params, build_graph, HIGH, LOW
from defaults import PROMPT, NEGATIVE_PROMPT

def test_two_stage_sampling_contract():
    p=Params('image.png',PROMPT,NEGATIVE_PROMPT)
    g=build_graph(p)
    a,b=g['sample_high']['inputs'],g['sample_low']['inputs']
    assert (a['start_at_step'],a['end_at_step'],b['start_at_step'],b['end_at_step'])==(0,3,3,8)
    assert a['add_noise']=='enable' and a['return_with_leftover_noise']=='enable'
    assert b['add_noise']=='disable' and b['return_with_leftover_noise']=='disable'
    assert b['latent_image']==['sample_high',0]
    assert g['high']['inputs']['unet_name']==HIGH and g['low']['inputs']['unet_name']==LOW
    assert g['conditioning']['inputs']['length']==81
    assert g['features']['inputs']['crop']=='none'

@pytest.mark.parametrize('kwargs', [{'width':191},{'num_frames':80},{'switch_step':8},{'switch_step':0}])
def test_reject_invalid_requests(kwargs):
    with pytest.raises(ValueError):build_graph(Params('image.png','p','',**kwargs))

def test_independent_high_low_controls():
    p=Params('image.png','p','n',steps=12,switch_step=5,cfg=2,low_cfg=3,shift=4,low_shift=6,seed=123)
    g=build_graph(p)
    assert g['sample_high']['inputs']['cfg']==2 and g['sample_low']['inputs']['cfg']==3
    assert g['high_shift']['inputs']['shift']==4 and g['low_shift']['inputs']['shift']==6
    assert g['sample_low']['inputs']['noise_seed']==123
