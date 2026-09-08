"""Public total Boolean output tests; each action returns exactly one bit."""
from .prediction import category

PROPERTIES = ('error', 'list', 'empty', 'positive_scalar', 'length_ge3',
              'has_negative', 'all_even_nonempty', 'palindrome_nonempty')


def observe(value, property_name):
    category(value)
    if property_name not in PROPERTIES:
        raise ValueError('unknown output property')
    if property_name == 'error':
        return value is None
    if value is None:
        return False
    if property_name == 'list':
        return type(value) is list
    if property_name == 'positive_scalar':
        return type(value) is int and value > 0
    if type(value) is not list:
        return False
    if property_name == 'empty':
        return len(value) == 0
    if property_name == 'length_ge3':
        return len(value) >= 3
    if property_name == 'has_negative':
        return any(x < 0 for x in value)
    if property_name == 'all_even_nonempty':
        return bool(value) and all(x%2 == 0 for x in value)
    return bool(value) and value == value[::-1]
