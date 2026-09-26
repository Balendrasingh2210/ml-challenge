"""Text normalization for business names and addresses."""

import re
import unicodedata

# Common business suffix abbreviations → canonical form
NAME_ABBREVS = {
    r'\bcorp\b': 'corporation',
    r'\binc\b': 'incorporated',
    r'\bllc\b': 'llc',
    r'\bllp\b': 'llp',
    r'\bltd\b': 'limited',
    r'\bpvt\b': 'private',
    r'\bco\b': 'company',
    r'\bcos\b': 'company',
    r'\bintl\b': 'international',
    r'\bsvcs\b': 'services',
    r'\bsvc\b': 'service',
    r'\bmgmt\b': 'management',
    r'\bmfg\b': 'manufacturing',
    r'\btech\b': 'technology',
    r'\btechs\b': 'technologies',
    r'\bassoc\b': 'associates',
    r'\bassy\b': 'associates',
    r'\bgrp\b': 'group',
    r'\bentps\b': 'enterprises',
    r'\bentp\b': 'enterprise',
    r'\benterp\b': 'enterprises',
    r'\bengr\b': 'engineering',
    r'\bengg\b': 'engineering',
    r'\binds\b': 'industries',
    r'\bind\b': 'industries',
    r'\bconst\b': 'construction',
    r'\bdist\b': 'distributors',
    r'\bdistrib\b': 'distributors',
    r'\bdev\b': 'development',
    r'\bdevs\b': 'developers',
    r'\bsol\b': 'solutions',
    r'\bsols\b': 'solutions',
    r'\bsys\b': 'systems',
    r'\bmedl\b': 'medical',
    r'\bmed\b': 'medical',
    r'\bfinl\b': 'financial',
    r'\bfin\b': 'financial',
    r'\bcomm\b': 'communications',
    r'\bcomms\b': 'communications',
    r'\bprop\b': 'properties',
    r'\bprops\b': 'properties',
    r'\brealty\b': 'realty',
    r'\bcons\b': 'consultants',
    r'\bconsult\b': 'consultants',
    r'&': 'and',
}

# Address abbreviations
ADDR_ABBREVS = {
    r'\bst\b': 'street',
    r'\bstr\b': 'street',
    r'\bave\b': 'avenue',
    r'\bav\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\brd\b': 'road',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bpkwy\b': 'parkway',
    r'\bhwy\b': 'highway',
    r'\bfwy\b': 'freeway',
    r'\bexpy\b': 'expressway',
    r'\btr\b': 'trail',
    r'\btrl\b': 'trail',
    r'\bcir\b': 'circle',
    r'\bsq\b': 'square',
    r'\bterr\b': 'terrace',
    r'\bter\b': 'terrace',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\bflr\b': 'floor',
    r'\bfl\b': 'floor',
    r'\bn\b': 'north',
    r'\bs\b': 'south',
    r'\be\b': 'east',
    r'\bw\b': 'west',
    r'\bne\b': 'northeast',
    r'\bnw\b': 'northwest',
    r'\bse\b': 'southeast',
    r'\bsw\b': 'southwest',
}

STOPWORDS = {
    'the', 'a', 'an', 'of', 'in', 'at', 'by', 'for', 'to', 'on', 'is',
    'and', 'or', 'with', 'as', 'into', 'near', 'behind', 'above', 'below',
    'no', 'not', 'null', 'none', 'nan',
}


def normalize_unicode(text: str) -> str:
    """Convert to ASCII by decomposing unicode characters."""
    if not isinstance(text, str):
        return ''
    # Normalize unicode (NFD decomposes accented chars)
    nfkd = unicodedata.normalize('NFKD', text)
    # Keep only ASCII printable
    ascii_text = nfkd.encode('ascii', 'ignore').decode('ascii')
    return ascii_text


def normalize_name(name: str) -> str:
    """Normalize a business name for comparison."""
    if not isinstance(name, str) or name.strip().lower() in ('', 'nan', 'none', 'null'):
        return ''
    text = name.lower()
    text = normalize_unicode(text)
    # Remove punctuation except hyphens within words
    text = re.sub(r"[^\w\s-]", ' ', text)
    text = re.sub(r'(?<!\w)-|-(?!\w)', ' ', text)  # remove standalone hyphens
    # Expand abbreviations
    for pattern, replacement in NAME_ABBREVS.items():
        text = re.sub(pattern, replacement, text)
    # Collapse whitespace
    text = ' '.join(text.split())
    return text


def normalize_address(addr: str) -> str:
    """Normalize a business address for comparison."""
    if not isinstance(addr, str) or addr.strip().lower() in ('', 'nan', 'none', 'null'):
        return ''
    text = addr.lower()
    text = normalize_unicode(text)
    text = re.sub(r"[^\w\s-]", ' ', text)
    text = re.sub(r'(?<!\w)-|-(?!\w)', ' ', text)
    for pattern, replacement in ADDR_ABBREVS.items():
        text = re.sub(pattern, replacement, text)
    text = ' '.join(text.split())
    return text


def get_name_tokens(norm_name: str) -> list[str]:
    """Get significant word tokens from a normalized name."""
    tokens = norm_name.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def get_char_ngrams(text: str, n: int = 3) -> set[str]:
    """Get character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def extract_address_numbers(addr: str) -> list[str]:
    """Extract numeric components from an address."""
    if not addr:
        return []
    return re.findall(r'\b\d+\b', addr)
