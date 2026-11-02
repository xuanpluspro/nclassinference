"""Strict restoration of native nclassbenchmark SMP checkpoints."""
from pathlib import Path

def load(weights, trust=False, config_path=None):
    import torch
    import segmentation_models_pytorch as smp
    import json
    with Path(weights).open('rb') as f:
        try: ck = torch.load(f, map_location='cpu', weights_only=not trust)
        except Exception as exc:
            raise RuntimeError('Checkpoint load failed. For your own trusted training checkpoint, add --trust-checkpoint. '+str(exc)) from exc
    if not isinstance(ck, dict): raise ValueError('Expected checkpoint dictionary')
    config = ck.get('config')
    if config_path:
        supplied = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
        if config is not None:
            for key in ('arch','encoder','num_classes','class_names'):
                if key in supplied and key in config and supplied[key] != config[key]: raise ValueError(f'Config conflicts with checkpoint: {key}')
        config = {**(config or {}), **supplied}
    if not isinstance(config,dict): raise ValueError('Checkpoint missing config; supply --model-config JSON with architecture and class schema')
    for k in ('arch','encoder','num_classes'): 
        if k not in config: raise ValueError('Missing model config: '+k)
    from common import validate_schema
    config = validate_schema(dict(config))
    state = ck.get('model', ck.get('state_dict'))
    if state is None and all(torch.is_tensor(v) for v in ck.values()): state = ck
    if not isinstance(state, dict) or not state: raise ValueError('Missing model state dictionary')
    if all(k.startswith('module.') for k in state): state = {k[7:]:v for k,v in state.items()}
    model = smp.create_model(config['arch'], encoder_name=config['encoder'], encoder_weights=None, in_channels=3, classes=config['num_classes'])
    model.load_state_dict(state, strict=True)
    return model, config
