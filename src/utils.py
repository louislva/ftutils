import re
import random

def str_to_filename(s):
    s = s.lower()
    s = re.sub(r'[^a-zA-Z0-9_\-]', '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s

def random_hex(n):
    return "".join(hex(random.randint(0, 15))[2:] for _ in range(n))