"""Injective string outcome encoding for future categorical-score adapters."""
import json


def category(value):
    if type(value) is not str:
        raise ValueError('string output required; None is not an empty string')
    return json.dumps(value,ensure_ascii=True,separators=(',',':'))
