"""Safe lexical splitting shared by prose, table rows and uncertain fallback text."""
import re

ATOM = r'[+\-\u2212]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-\u2212]?\d+)?'
NUMERIC = re.compile(r'(?<!\w)' + ATOM + r'(?:\s*/\s*' + ATOM + r')*(?:%|\s*(?:MPa|GPa|kPa|Pa|kN/mm|kN|kJ|mm|cm|km|m|kg|mg|Hz|ms|s|N|K|\u00b0C)(?!\w)|[^\W\d_][\w/^\u00b7\u00b2\u00b3]*)?(?!\w)')

def safe_boundaries(text):
    protected = [(m.start(), m.end()) for m in NUMERIC.finditer(text)]
    return [i for i in range(1, len(text)+1) if not any(a < i < b for a,b in protected)]

def split_safe(text, fits):
    """Exact source slices: whitespace first, then lexical, then numeric-safe characters."""
    start = 0
    boundaries = safe_boundaries(text)
    while start < len(text):
        if fits(text[start:]):
            yield start, len(text)
            return
        candidates = [i for i in boundaries if i > start]
        stop = start
        for end in candidates:
            if fits(text[start:end]):
                stop = end
            else:
                break
        if stop == start:
            raise ValueError('Indivisible numeric expression or required context cannot fit the effective token limit.')
        whitespace = [i for i in candidates if i <= stop and text[i-1].isspace() and text[start:i].strip()]
        lexical = [i for i in candidates if i <= stop and text[i-1] in ',;|/']
        stop = max(whitespace or lexical or [stop])
        yield start, stop
        start = stop
