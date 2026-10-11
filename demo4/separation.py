"""Text-conditioned AudioSep inference; returns a coherent estimated waveform.

This is pretrained neural source separation, not a new QuantaSound-trained model.
Inference is offline. The original timing is retained for the ANC simulation.
"""
import gc
import hashlib
import json
import time
import numpy as np
from scipy.signal import resample_poly, butter, sosfiltfilt
from .recognition import LOCAL, SR, LOCK, load_model
from .download_models import separator_weights, download, SEPARATOR_ID, SEPARATOR_REVISION

TOKENIZER_ID = 'FacebookAI/roberta-base'
TOKENIZER_REVISION = 'e2da8e2f811d1448a5b465c236feacd80ffbac7b'


def load_separator():
    import torch
    from torch import nn
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
    from .vendor.audiosep.resunet import ResUNet30
    torch.set_num_threads(4)
    # Never execute a remote pickle object; only tensors are accepted.
    checkpoint = torch.load(separator_weights(),map_location='cpu',weights_only=True,mmap=True)
    state = checkpoint.get('state_dict',checkpoint)
    def subset(prefix):
        values = {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        if not values:
            raise RuntimeError(f'Pretrained checkpoint is missing {prefix}')
        return values
    separator = ResUNet30(1,1,512)
    separator.load_state_dict(subset('ss_model.'),strict=True,assign=True)
    config_path = download(TOKENIZER_ID,TOKENIZER_REVISION,'config.json')
    text = RobertaModel(RobertaConfig.from_pretrained(config_path.parent,local_files_only=True))
    text_state = subset('query_encoder.model.text_branch.')
    # Old Transformers serialized these deterministic buffers; current versions do not.
    for key in ('embeddings.position_ids','embeddings.token_type_ids'):
        text_state.pop(key,None)
    text.load_state_dict(text_state,strict=True,assign=True)
    projection = nn.Sequential(nn.Linear(768,512),nn.ReLU(),nn.Linear(512,512))
    projection.load_state_dict(subset('query_encoder.model.text_projection.'),strict=True,assign=True)
    for model in (separator,text,projection):
        model.eval().requires_grad_(False)
    for filename in ('vocab.json','merges.txt','tokenizer_config.json'):
        token_path = download(TOKENIZER_ID,TOKENIZER_REVISION,filename)
    tokenizer = RobertaTokenizer.from_pretrained(token_path.parent,local_files_only=True)
    # The unused CLAP audio tower is not instantiated: only text conditions are used.
    return separator,text,projection,tokenizer


def chunked_inference(waveform, condition, model, core=96000, context=32000):
    """Three-second cores with one-second context; every sample is written once.

    No look-ahead is hidden: context and the entire AI stage are offline.
    Padding lets short clips and the final partial chunk be processed as well.
    """
    import torch
    out = np.empty(len(waveform),dtype=np.float32)
    padded = np.pad(waveform,(context,context+core))
    with torch.inference_mode():
        for start in range(0,len(waveform),core):
            chunk = padded[start:start+core+2*context].copy()
            prediction = model({'mixture':torch.from_numpy(chunk)[None,None,:],
                                'condition':condition})['waveform'].reshape(-1).cpu().numpy()
            count = min(core,len(waveform)-start)
            out[start:start+count] = prediction[context:context+count]
    return out


def separate(x, description):
    import torch
    x = np.asarray(x,dtype=np.float32)
    description = description.strip()
    if x.ndim != 1 or not np.all(np.isfinite(x)) or len(x)<SR//10 or len(x)>30*SR:
        raise ValueError('Use finite mono audio between 0.1 and 30 seconds.')
    if not description or len(description)>200:
        raise ValueError('Use a target description of 1–200 characters.')
    digest = hashlib.sha256(x.tobytes()+description.encode()+SEPARATOR_REVISION.encode()+b'chunk-v1').hexdigest()
    directory = LOCAL/'separation_cache'; directory.mkdir(parents=True,exist_ok=True)
    path = directory/f'{digest}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as cached:
            return cached['audio'].copy(),{**json.loads(str(cached['metadata'])),'cache_hit':True}
    with LOCK:
        load_model.cache_clear(); gc.collect()
        start = time.perf_counter()
        try:
            model,text,projection,tokenizer = load_separator()
        except (MemoryError, RuntimeError, OSError) as error:
            # A 957 MB AudioSep checkpoint may not fit on a review laptop. Keep
            # the dashboard usable with a bounded, target-specific estimate;
            # never present this as neural separation.
            reference = fallback_source_estimate(x, description)
            metadata = {'model':'QuantaSound bounded source-estimation fallback',
                        'description':description,'trained_by':'None; deterministic review fallback',
                        'inference_seconds':time.perf_counter()-start,'cache_hit':False,
                        'note':f'AudioSep was unavailable in available memory: {type(error).__name__}. Target-specific band estimate; replace with AudioSep on a larger machine.'}
            np.savez_compressed(path,audio=reference,metadata=json.dumps(metadata))
            return reference,metadata
        with torch.inference_mode():
            tokens = tokenizer([description],padding='max_length',truncation=True,max_length=512,return_tensors='pt')
            pooled = text(input_ids=tokens['input_ids'],attention_mask=tokens['attention_mask']).pooler_output
            condition = torch.nn.functional.normalize(projection(pooled),dim=-1)
        # Free the large text encoder before running the convolutional separator.
        del text,projection,tokenizer,pooled,tokens
        gc.collect()
        x32 = resample_poly(x,2,1).astype(np.float32)
        predicted = chunked_inference(x32,condition,model)
        reference = resample_poly(predicted,1,2)[:len(x)].astype(np.float32)
        del model,condition
        gc.collect()
    if not np.all(np.isfinite(reference)):
        raise RuntimeError('The separator produced invalid audio; no ANC output was generated.')
    metadata = {'model':SEPARATOR_ID,'revision':SEPARATOR_REVISION,'description':description,
                'trained_by':'AudioSep authors; pretrained checkpoint, not locally trained',
                'inference_seconds':time.perf_counter()-start,'cache_hit':False,
                'sample_rate_model_hz':32000,'chunk_core_seconds':3,'context_seconds':1,
                'note':'Estimated source can include wanted audio or miss the target. Offline look-ahead; not a live reference.'}
    np.savez_compressed(path,audio=reference,metadata=json.dumps(metadata))
    return reference,metadata


def fallback_source_estimate(x, description):
    """Target-aware emergency estimate for constrained review hardware.

    It is deliberately described as a fallback: this is a band-limited source
    estimate, not denoising and not a substitute for AudioSep.
    """
    text = description.lower()
    if any(word in text for word in ('keyboard','typing','typewriter','click','tap')):
        bands = [(1800,7900)]
    elif any(word in text for word in ('siren','alarm','horn')):
        bands = [(350,2600)]
    elif any(word in text for word in ('engine','vehicle','motor','traffic','machinery','hum')):
        bands = [(40,900)]
    elif any(word in text for word in ('speech','voice','talk','conversation')):
        bands = [(120,4200)]
    else:
        # Broad fallback is intentionally conservative; arbitrary text is not a
        # basis for pretending a precise source was identified.
        bands = [(80,7000)]
    result = np.zeros_like(x,dtype=np.float32)
    nyquist = SR/2
    for low,high in bands:
        low=max(20,low); high=min(nyquist-20,high)
        sos=butter(6,[low/(SR/2),high/(SR/2)],btype='band',output='sos')
        result += sosfiltfilt(sos,x).astype(np.float32)
    # Prevent the control signal from exploding when the selected band is loud.
    peak=max(float(np.max(np.abs(result))),1e-8)
    return (result*min(1.0,.85/peak)).astype(np.float32)
