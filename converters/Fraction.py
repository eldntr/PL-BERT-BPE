from singleton_decorator import singleton
import re
from .Cardinal import Cardinal

@singleton
class Fraction:
    """
    Class ini mengkonversi pecahan ke dalam bentuk terucap Bahasa Indonesia.
    
    Logika utama:
    - 1/2 -> "setengah"
    - 1/4 -> "seperempat"
    - 1/N -> "seper" + cardinal(N) (misal: "sepertiga", "seperdelapan")
    - M/N -> cardinal(M) + " per" + cardinal(N) (misal: "dua pertiga")
    - X Y/Z -> cardinal(X) + (hasil konversi Y/Z) (misal: "delapan setengah")
    """
    def __init__(self):
        super().__init__()
        # Regex
        self.filter_regex = re.compile(",")
        self.space_filter_regex = re.compile(" ")
        self.nbsp_regex = re.compile("\u00A0")
        
        # Dict untuk karakter unicode pecahan
        self.trans_dict = {
            "½": "setengah",
            "⅓": "sepertiga",
            "⅔": "dua pertiga",
            "¼": "seperempat",
            "¾": "tiga perempat",
            "⅕": "seperlima",
            "⅖": "dua perlima",
            "⅗": "tiga perlima",
            "⅘": "empat perlima",
            "⅙": "seperenam",
            "⅚": "lima perenam",
            "⅛": "seperdelapan",
            "⅜": "tiga perdelapan",
            "⅝": "lima perdelapan",
            "⅞": "tujuh perdelapan",
        }
        
        # Regex untuk karakter unicode (di-build dari dict di atas)
        self.special_regex = re.compile(f"({'|'.join(self.trans_dict.keys())})")
        
        # Konverter Cardinal
        self.cardinal = Cardinal()

        # Dict 'trans_denominator' dan 'edge_dict' (versi Inggris)
        # tidak digunakan untuk logika Bahasa Indonesia.

    def _normalize_digits(self, value: str) -> str:
        # Hilangkan pemisah ribuan atau karakter non-digit umum
        return value.replace(".", "").replace(",", "")

    def _clean_spaces(self, token: str) -> str:
        token = self.nbsp_regex.sub(" ", token)
        return self.space_filter_regex.sub("", token.strip())

    def _parse_slash_fraction(self, token: str):
        if "/" not in token:
            return None

        numerator_raw, denominator_raw = token.split("/", 1)
        numerator = self._clean_spaces(numerator_raw)
        denominator = self._clean_spaces(denominator_raw)

        if not numerator or not denominator:
            return None

        if "-" in numerator[1:]:
            return None

        numerator_digits = self._normalize_digits(numerator.lstrip("-"))
        if not numerator_digits or not numerator_digits.isdigit():
            return None

        denominator_negative = denominator.startswith("-")
        if denominator_negative:
            denominator = denominator[1:]
        if not denominator:
            return None

        denominator_parts = denominator.split("-")
        if any(part == "" for part in denominator_parts):
            return None

        denominator_digits = []
        for part in denominator_parts:
            digits = self._normalize_digits(part)
            if not digits or not digits.isdigit():
                return None
            denominator_digits.append(digits)

        return {
            "numerator": numerator,
            "numerator_digits": numerator_digits,
            "numerator_negative": numerator.startswith("-"),
            "denominator_parts": denominator_parts,
            "denominator_digits": denominator_digits,
            "denominator_negative": denominator_negative,
        }

    def _render_denominator(self, parts, is_negative):
        if len(parts) == 1:
            text = self.cardinal.convert(parts[0])
        else:
            connector = "sampai" if len(parts) == 2 else "strip"
            words = []
            for idx, part in enumerate(parts):
                if idx > 0:
                    words.append(connector)
                words.append(self.cardinal.convert(part))
            text = " ".join(words)

        if is_negative:
            text = f"minus {text}".strip()
        return text.strip()

    def looks_like_fraction(self, token: str) -> bool:
        cleaned = self.filter_regex.sub("", token or "")
        cleaned = self.nbsp_regex.sub(" ", cleaned)
        if not cleaned:
            return False
        if self.special_regex.search(cleaned):
            return True
        return self._parse_slash_fraction(cleaned) is not None

    def convert(self, token: str) -> str:
        # 1 Filter koma
        token = self.filter_regex.sub("", token)
        token = self.nbsp_regex.sub(" ", token)
        
        # 2 Cek kasus spesial unicode (misal: ½)
        match = self.special_regex.search(token)
        if match:
            # 3 Ambil teks pecahan (misal: "setengah")
            frac = match.group(1)
            frac_text = self.trans_dict[frac]

            # 4 Cek sisa angka (misal: "8" dari "8 ½")
            remainder = self.special_regex.sub("", token).strip()
            
            # 5 Jika ada sisa, konversi ke cardinal dan gabungkan
            if remainder:
                prefix = self.cardinal.convert(remainder)
                # "delapan" + "setengah" -> "delapan setengah"
                result = f"{prefix} {frac_text}"
            else:
                # Jika tidak, kembalikan teks pecahan
                result = frac_text
        
        else:
            parsed = self._parse_slash_fraction(token)
            if parsed:
                result = self._render_fraction(parsed)
            else:
                # Kasus tidak terduga
                result = token

        return result

    def _render_fraction(self, parsed_parts) -> str:
        numerator_text = self.cardinal.convert(parsed_parts["numerator"])
        denominator_text = self._render_denominator(
            parsed_parts["denominator_parts"],
            parsed_parts["denominator_negative"],
        )

        simple_denominator = (
            not parsed_parts["denominator_negative"]
            and len(parsed_parts["denominator_parts"]) == 1
        )

        numerator_digits = parsed_parts["numerator_digits"].lstrip("0") or "0"
        denominator_digits = parsed_parts["denominator_digits"][0].lstrip("0") or "0"

        if simple_denominator and numerator_digits == "1":
            if denominator_digits == "2":
                base = "setengah"
            elif denominator_digits == "4":
                base = "seperempat"
            else:
                base = f"seper{denominator_text}"
            prefix = "minus " if parsed_parts["numerator_negative"] else ""
            return f"{prefix}{base}".strip()

        return f"{numerator_text} per {denominator_text}"
