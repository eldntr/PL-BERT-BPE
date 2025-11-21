import pandas as pd
import unicodedata
import re

from converters.Plain      import Plain
from converters.Punct      import Punct
from converters.Date       import Date
from converters.Letters    import Letters
from converters.Cardinal   import Cardinal
from converters.Verbatim   import Verbatim
from converters.Decimal    import Decimal
from converters.Measure    import Measure
from converters.Money      import Money
from converters.Ordinal    import Ordinal
from converters.Time       import Time
from converters.Electronic import Electronic
from converters.Digit      import Digit
from converters.Fraction   import Fraction
from converters.Telephone  import Telephone
from converters.Address    import Address
from converters.Roman    import Roman
from converters.Range    import Range


months = [
    'jan',
    'feb',
    'mar',
    'apr',
    'mei',
    'jun',
    'jul',
    'agu',
    'sep',
    'okt',
    'nov',
    'des',
    'januari',
    'februari',
    'maret',
    'april',
    'mei',
    'juni',
    'juli',
    'agustus',
    'september',
    'oktober',
    'november',
    'desember'
]

era_suffixes = {"sm", "bc", "ad", "m"}
equals_tokens = {"=", "=="}
noise_symbol_chars = set("#*@^~`|\\")
power_unit_map = {
    "s": "detik",
    "ms": "milidetik",
    "m": "meter",
}
unit_token_map = {
    "ms": "milidetik",
    "s": "detik",
    "m": "meter",
}
time_zones = {"WIB", "WITA", "WIT"}
power_token_pattern = re.compile(
    r"^(?P<value>\d+)(?:\s*(?:\^(?P<exp>[-+]?\d+)|([−-](?P<exp_alt>\d+))))(?P<unit>[A-Za-zµμ]*)?$"
)
time_pattern = re.compile(r"^\d{1,2}\.\d{2}$")
SMART_PUNCT_TRANSLATION = str.maketrans({
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "’": "'",
    "‘": "'",
    "‚": "'",
    "‛": "'",
    "—": "-",
    "–": "-",
    "−": "-",
})

labels = {
    "PLAIN": Plain(),
    "PUNCT": Punct(),
    "DATE": Date(),
    "LETTERS": Letters(),
    "CARDINAL": Cardinal(),
    "VERBATIM": Verbatim(),
    "DECIMAL": Decimal(),
    "MEASURE": Measure(),
    "MONEY": Money(),
    "ORDINAL": Ordinal(),
    "TIME": Time(),
    "ELECTRONIC": Electronic(),
    "DIGIT": Digit(),
    "FRACTION": Fraction(),
    "TELEPHONE": Telephone(),
    "ADDRESS": Address(),
    "ROMAN": Roman(),
    "RANGE": Range()
}

def split_given_size(sequence, size):
    for idx in range(0, len(sequence), size):
        yield sequence[idx:idx + size]


def word_tokenize(text):
    return text.split()


def detokenize(tokens):
    return " ".join(tokens)


WRAP_PREFIX_CHARS = "(['\"“”‘’[{"
WRAP_SUFFIX_CHARS = ")]}'\"“”‘’.,;:!?"

def strip_parentheses(token: str):
    if token is None:
        return "", "", ""

    prefix = ""
    suffix = ""
    core = token

    while core and core[0] in WRAP_PREFIX_CHARS:
        prefix += core[0]
        core = core[1:]

    while core and core[-1] in WRAP_SUFFIX_CHARS:
        suffix = core[-1] + suffix
        core = core[:-1]

    return prefix, core, suffix

def normalize_split(text):
    words = word_tokenize(text)
    normalized_chunks = []
    for chunk in split_given_size(words, 500):
        chunk_text = detokenize(chunk)
        normalized_chunks.append(normalize_text(chunk_text))
    return " ".join(filter(None, (chunk.strip() for chunk in normalized_chunks)))

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def is_oridinal(inputString):
    lowered = inputString.lower()
    if lowered.endswith(("th", "nd", "st", "rd")):
        return True
    if lowered.startswith("ke-"):
        return lowered[3:].isdigit()
    if lowered.startswith("ke"):
        return lowered[2:].isdigit()
    return False

def is_money(inputString):
    stripped = inputString.strip()
    if stripped.startswith(('$', '€', '£', '¥')):
        return True
    upper = stripped.upper()
    return upper.startswith(("RP", "IDR"))

def is_time(inputString):
    return ":" in inputString

def is_cardinal(inputString):
    cleaned = inputString.replace(".", "").replace(",", "").replace(" ", "")
    return cleaned.isdigit()


def match_power_token(token):
    normalized = token.replace("−", "-").replace("--", "-")
    if "^" not in normalized:
        return None
    match = power_token_pattern.match(normalized)
    if match:
        value = match.group("value")
        exp = match.group("exp")
        exp_alt = match.group("exp_alt")
        if exp is None and exp_alt is None:
            return None

        sign = ""
        if exp is not None:
            sign = "minus " if exp.startswith("-") else "plus " if exp.startswith("+") else ""
            exp_value = exp.lstrip("+-")
        else:
            exp_value = exp_alt
            sign = "minus "

        if not exp_value or not exp_value.isdigit():
            return None

        unit = match.group("unit") or ""
        unit_lower = unit.lower()
        unit_text = power_unit_map.get(unit_lower) if unit_lower else None

        exp_phrase = f"pangkat {sign}{labels['CARDINAL'].convert(exp_value)}".strip()
        return value, exp_phrase, unit_text
    return None

def is_fraction(inputString):
    return labels["FRACTION"].looks_like_fraction(inputString)

def is_decimal(inputString):
    return bool(re.match(r"^-?\d+[.,]\d+$", inputString))

def is_range(inputString) : 
    return "-" in inputString and "/" not in inputString

def is_url(inputString):
    return "//" in inputString or ".com" in inputString or ".html" in inputString

def has_month(inputString):
    return inputString.lower() in months

def is_era(token):
    code = clean_era_code(token)
    if code not in era_suffixes:
        return False
    stripped = re.sub(r"[.\s]", "", token)
    return stripped.upper() == stripped and stripped != ""

def clean_era_code(token):
    return re.sub(r"[^a-z]", "", token.lower())

def is_roman_numeral(token):
    if not token:
        return False
    stripped = token.strip(".")
    if not stripped or not stripped.isupper():
        return False
    return labels["ROMAN"].check_if_roman(stripped)

def is_equals(token):
    return token.strip() in equals_tokens


def is_noise_symbol(token):
    if not token:
        return False
    if any(char in noise_symbol_chars for char in token):
        return True
    return bool(re.fullmatch(r"[^\w\s]{2,}", token))


def is_pure_punctuation(text: str) -> bool:
    if not text:
        return True
    cleaned = re.sub(r"\s+", "", text)
    if not cleaned:
        return True
    if any(char.isalnum() for char in cleaned):
        return False
    allowed_keep = {".", ",", "!", "?", ";", ":"}
    if cleaned in allowed_keep:
        return False
    return True

def normalize_single(text, prev_text = "", next_text = ""):
    prefix, token, suffix = strip_parentheses(text)
    _, prev_token, _ = strip_parentheses(prev_text)
    _, next_token, _ = strip_parentheses(next_text)

    power_match = match_power_token(token)
    if power_match:
        value_text = labels['CARDINAL'].convert(power_match[0])
        exp_text = power_match[1]
        unit_text = power_match[2]
        token = f"{value_text} {exp_text}"
        if unit_text:
            token = f"{token} {unit_text}"
    elif is_url(token):
        token = labels['ELECTRONIC'].convert(token).upper()
    elif has_numbers(token):
        try:
            if has_month(prev_token):
                prev_month = labels['DATE'].get_month(prev_token.lower())
                token = labels['DATE'].convert(f"{prev_month} {token}").replace(prev_month, "").strip()
            elif has_month(next_token):
                next_month = labels['DATE'].get_month(next_token.lower())
                # Date converter expects the month before the year,
                # so feed it as "<month> <year>" to avoid misparsing the month as an era suffix.
                token = labels['DATE'].convert(f"{next_month} {token}").replace(next_month, "").strip()
            elif prev_token.lower() == "pukul" and time_pattern.match(token):
                token = labels['TIME'].convert(token)
            elif next_token.upper() in time_zones and time_pattern.match(token):
                token = labels['TIME'].convert(token)
            elif is_oridinal(token):
                token = labels['ORDINAL'].convert(token)
            elif is_time(token):
                token = labels['TIME'].convert(token)
            elif is_money(token):
                token = labels['MONEY'].convert(token)
            elif is_fraction(token):
                token = labels['FRACTION'].convert(token)
            elif is_decimal(token):
                token = labels['DECIMAL'].convert(token)
            elif is_cardinal(token):
                token = labels['CARDINAL'].convert(token)
            elif is_range(token):
                token = labels['RANGE'].convert(token)
            else:
                token = labels['DATE'].convert(token)
        except Exception:
            token = labels['CARDINAL'].convert(token)

        if has_numbers(token):
            token = labels['CARDINAL'].convert(token)
    elif token == "#" and has_numbers(next_token):
        token = "number"
    elif has_month(token):
        token = labels['DATE'].get_month(token.lower())
    elif is_era(token):
        token = labels['DATE'].get_suffix(clean_era_code(token))
    elif is_equals(token):
        token = "sama dengan"
    elif is_noise_symbol(token):
        return ""
    elif token in unit_token_map:
        token = unit_token_map[token]
    elif token.islower() and token.lower() in unit_token_map:
        token = unit_token_map[token.lower()]
    elif is_roman_numeral(token):
        number_value, suffix = labels["ROMAN"].convert(token)
        token = labels["CARDINAL"].convert(number_value)
        if suffix:
            token = f"{token} {suffix}"

    normalized_token = (prefix + token + suffix).replace("$", "")
    if is_pure_punctuation(normalized_token):
        return ""
    return normalized_token

def normalize_text(text):
    text = remove_accents(text).translate(SMART_PUNCT_TRANSLATION)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)

    def collapse_number_groups(match):
        candidate = match.group(0)
        return candidate.replace(" ", "")

    text = re.sub(r"\b\d{1,3}(?: \d{3})+\b", collapse_number_groups, text)
    words = word_tokenize(text)

    df = pd.DataFrame(words, columns=['before'])

    df['after'] = df['before']
    
    df['previous'] = df.before.shift(1).fillna('') + "|" + df.before + "|" + df.before.shift(-1).fillna('')
    
    df['after'] = df['previous'].apply(lambda m: normalize_single(m.split('|')[1], m.split('|')[0], m.split('|')[2]))
    
    normalized = detokenize(df['after'].tolist()).replace("’ s", "'s").replace(" 's", "'s")
    return normalized.strip()

if __name__ == '__main__' : 
    text = 'hello (23 Jan 2020, 12:10 AM)'
    out = normalize_text(text)
    print(out)
