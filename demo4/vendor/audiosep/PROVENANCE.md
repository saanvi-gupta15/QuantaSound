Vendored from https://github.com/Audio-AGI/AudioSep at commit
944583f18b84589dc965de3ad77525c945334252 (MIT, see LICENSE).

Only `base.py` and `resunet.py` are used. The base import in resunet.py
is relative. QuantaSound implements its own bounded chunk inference;
the upstream chunk_inference method is not called. The text encoder
uses the checkpoint's RoBERTa and projection weights, with the same
pooler, ReLU projection and L2 normalization as upstream CLAP.

Weights: https://huggingface.co/nielsr/audiosep-demo at revision
9188caa202a0d31845367e4dad8bb5549eddb4ba, recommended by the upstream
README. These are pretrained research weights, not QuantaSound training.
