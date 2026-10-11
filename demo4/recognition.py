"""Real AudioSet-trained AST inference. No hand-written class rules or fallback labels."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import threading
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / 'local'
MODEL_ID = 'MIT/ast-finetuned-audioset-10-10-0.4593'
REVISION = 'f826b80d28226b62986cc218e5cec390b1096902'
SR = 16000
LOCK = threading.Lock()


def read_audio(path, max_seconds=30):
    """Read supported soundfile formats, reject oversize inputs; no silent truncation."""
    info = sf.info(path)
    if info.duration > max_seconds:
        raise ValueError(f'Use a clip no longer than {max_seconds} seconds for this CPU demo.')
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if len(x) < int(.1*sr) or not np.all(np.isfinite(x)):
        raise ValueError('Use a finite audio clip at least 0.1 seconds long.')
    if sr != SR:
        factor = gcd(sr, SR)
        x = resample_poly(x, SR//factor, sr//factor)
    return np.asarray(x, dtype=np.float32)


@lru_cache(maxsize=1)
def load_model():
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    torch.set_num_threads(4)
    from .download_models import classifier_directory
    directory = classifier_directory()
    feature = ASTFeatureExtractor.from_pretrained(directory, local_files_only=True)
    model = ASTForAudioClassification.from_pretrained(directory,
                local_files_only=True, use_safetensors=True).eval()
    return feature, model


def analyze(x):
    """Independent sigmoid scores across four-second windows, NOT softmax classes.

    Peak window scores retain shorter events in longer clips. Scores are model
    outputs, not calibrated certainty; overlap, masking and unknown sounds remain.
    """
    import torch
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError('Expected a finite mono reference.')
    digest = hashlib.sha256(x.tobytes()+REVISION.encode()).hexdigest()
    cache_path = LOCAL/'recognition_cache'/f'{digest}.json'
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding='utf-8'))
    with LOCK:
        feature, model = load_model()
        windows, intervals = [], []
        window_samples = 4*SR
        for start in range(0, len(x), window_samples):
            chunk = x[start:start+window_samples]
            if len(chunk) < SR:
                chunk = np.pad(chunk, (0, SR-len(chunk)))
            inputs = feature(chunk, sampling_rate=SR, return_tensors='pt')
            with torch.inference_mode():
                scores = torch.sigmoid(model(**inputs).logits)[0].cpu().numpy()
            windows.append(scores)
            intervals.append([start/SR, min(start+window_samples,len(x))/SR])
        scores = np.max(windows, axis=0)
        labels = [model.config.id2label[i] for i in range(len(scores))]
    ranking = np.argsort(scores)[::-1]
    output = {'model':MODEL_ID, 'revision':REVISION, 'class_count':len(labels),
              'trained_by':'MIT, pretrained/fine-tuned on AudioSet; not trained from scratch by QuantaSound',
              'duration_s':len(x)/SR, 'aggregation':'maximum independent sigmoid score across 4 s windows',
              'scores':{labels[i]:float(scores[i]) for i in ranking},
              'windows':[{'interval_s':interval, 'top':{labels[i]:float(values[i])
                         for i in np.argsort(values)[::-1][:5]}} for interval,values in zip(intervals,windows)],
              'note':'Candidate labels can overlap and can be wrong. Not all possible sounds are covered.'}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
    return output


def candidates(result, threshold=.035, limit=15):
    return [(label,score) for label,score in result['scores'].items() if score >= threshold][:limit]


if __name__ == '__main__':
    feature, model = load_model()
    print(f'Loaded {MODEL_ID} at {REVISION}: {len(model.config.id2label)} classes')
