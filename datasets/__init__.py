"""Dataset registry: builds the evaluation dataset specified by name."""

from .refcoco import build as build_refcoco

_BUILDERS = {
    'refcoco': build_refcoco,
    'refcoco+': build_refcoco,
    'refcocog': build_refcoco,
}


def build_dataset(dataset: str, args: object):
    if dataset not in _BUILDERS:
        raise ValueError(f'Unknown dataset: {dataset}. '
                         f'Supported: {list(_BUILDERS.keys())}')
    return _BUILDERS[dataset](args)
