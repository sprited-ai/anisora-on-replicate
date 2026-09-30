"""One bounded hosted test. Never automatically retries prediction creation."""
import argparse
import base64
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--version')
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--token-file',type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    token=args.token_file.read_text().strip()
    base='https://api.replicate.com/v1'
    def request(path, data=None, headers=None, timeout=60):
        req=urllib.request.Request(base+path,data=data,headers={
            'Authorization':'Bearer '+token, 'User-Agent':'Sprute-endpoint-validation/1.0', **(headers or {})})
        with urllib.request.urlopen(req,timeout=timeout) as response:
            return json.load(response)
    def upload(path):
        path=Path(path); boundary=uuid.uuid4().hex
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="content"; filename="{path.name}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()+path.read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
        return request('/files',body,{'Content-Type':f'multipart/form-data; boundary={boundary}'},timeout=180)['urls']['get']
    def prepare(value):
        if isinstance(value,dict) and set(value)=={'file'}:return upload(value['file'])
        if isinstance(value,list):return [prepare(x) for x in value]
        if isinstance(value,dict):return {k:prepare(v) for k,v in value.items()}
        return value
    version=args.version or request('/models/'+args.model)['latest_version']['id']
    inp=prepare(json.loads(args.inputs.read_text()))
    (args.out/'request.json').write_text(json.dumps({'version':version,'input':inp},indent=2))
    prediction=None; terminal=False
    started=time.monotonic()
    try:
        prediction=request('/predictions',json.dumps({'version':version,'input':inp}).encode(),
                           {'Content-Type':'application/json','Cancel-After':'10m'},timeout=90)
        pid=prediction['id'];print('prediction',pid,flush=True)
        last=None
        while True:
            (args.out/'prediction.json').write_text(json.dumps(prediction))
            status=prediction['status']
            if status!=last: print(status,round(time.monotonic()-started,1),flush=True);last=status
            if status in ('succeeded','failed','canceled','aborted'):
                terminal=True;break
            if time.monotonic()-started>640:raise TimeoutError('Hosted test exceeded 640 seconds')
            time.sleep(10)
            prediction=request('/predictions/'+pid)
        print('metrics',prediction.get('metrics'),flush=True)
        if status!='succeeded':raise RuntimeError(str(prediction.get('error'))+'\n'+(prediction.get('logs') or '')[-3000:])
        for key,value in prediction['output'].items():
            if not isinstance(value,str) or not (value.startswith('https://') or value.startswith('data:')):continue
            suffix={'video':'.mp4','frames':'.zip','metadata':'.json','reference_mask':'.png','driving_mask':'.mkv'}.get(key,'.bin')
            if value.startswith('data:'): content=base64.b64decode(value.split(',',1)[1])
            else:
                with urllib.request.urlopen(value,timeout=120) as response:content=response.read()
            (args.out/(key+suffix)).write_bytes(content)
    finally:
        if prediction and not terminal:
            request('/predictions/'+prediction['id']+'/cancel',b'',timeout=30)

if __name__=='__main__':main()
