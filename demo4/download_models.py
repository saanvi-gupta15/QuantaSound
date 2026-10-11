"""Download pinned pretrained weights locally; never commit model binaries."""
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
from filelock import FileLock
from .recognition import LOCAL, MODEL_ID, REVISION

SEPARATOR_ID = 'nielsr/audiosep-demo'
SEPARATOR_REVISION = '9188caa202a0d31845367e4dad8bb5549eddb4ba'


def separator_weights():
    return download(SEPARATOR_ID, SEPARATOR_REVISION, 'pytorch_model.bin',
                    '37f1691fb067e2575f1ad1cfbfe44b7b3da18e52f33fcb2b0937b72952f11ba1')


def download(model, revision, filename, sha256=None):
    path = LOCAL/'weights'/model.replace('/','--')/revision/filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path)+'.lock'):
        return _download(model,revision,filename,sha256)


def _download(model, revision, filename, sha256=None):
    """Plain HTTPS fallback for Windows networks where the HF transfer client stalls."""
    path = LOCAL/'weights'/model.replace('/','--')/revision/filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    temporary = path.with_suffix(path.suffix+'.part')
    digest = hashlib.sha256()
    url = f'https://huggingface.co/{model}/resolve/{revision}/{filename}'
    if sha256:
        # Bounded ranges also work through proxies that stall large streaming GETs.
        with requests.get(url,headers={'Range':'bytes=0-0'},timeout=(20,60)) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError('Model server did not honor a bounded range request.')
            size = int(response.headers['Content-Range'].split('/')[-1])
        block_size = 1024*1024
        lock = threading.Lock()
        with temporary.open('wb') as handle:
            handle.truncate(size)
            def transfer(start):
                end = min(start+block_size,size)-1
                for attempt in range(3):
                    try:
                        with requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,90)) as part:
                            part.raise_for_status()
                            content = part.content
                            if part.status_code != 206 or len(content) != end-start+1 or part.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                                raise RuntimeError('Invalid model range response.')
                        with lock:
                            handle.seek(start); handle.write(content)
                        return
                    except (requests.RequestException,RuntimeError):
                        if attempt == 2: raise
            with ThreadPoolExecutor(max_workers=12) as pool:
                for index,_ in enumerate(pool.map(transfer,range(0,size,block_size)),1):
                    if index % 64 == 0:
                        print(f'{filename}: {min(index*block_size,size)//1048576}/{size//1048576} MiB',flush=True)
        with temporary.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
        if digest.hexdigest() != sha256:
            raise RuntimeError(f'Model checksum failed: {filename}')
        temporary.replace(path)
        return path
    with requests.get(url,
                      stream=True, timeout=(20,120)) as response:
        response.raise_for_status()
        with temporary.open('wb') as handle:
            for block in response.iter_content(1024*1024):
                handle.write(block)
                digest.update(block)
    if sha256 and digest.hexdigest() != sha256:
        raise RuntimeError(f'Model checksum failed: {filename}')
    temporary.replace(path)
    return path


def classifier_directory():
    for filename in ('config.json','preprocessor_config.json','model.safetensors'):
        path = download(MODEL_ID, REVISION, filename,
            'ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d'
            if filename == 'model.safetensors' else None)
    return path.parent


if __name__ == '__main__':
    import sys
    if '--separator' in sys.argv:
        print(separator_weights(), flush=True)
    else:
        print(classifier_directory(), flush=True)
