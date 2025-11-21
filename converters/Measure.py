from singleton_decorator import singleton
import re
from .Decimal import Decimal
from .Fraction import Fraction

@singleton
class Measure:
    """
    Versi Bahasa Indonesia dari converter 'Measure'.
    
    Perubahan:
    - Menerjemahkan semua unit di 'prefixable_trans_dict' dan 'custom_dict'.
    - Menghapus pluralisasi (misal: "meters" -> "meter")
    - Memperbarui 'value_regex' dengan skala Indonesia (juta, miliar, dst.)
    - Menghapus logika "of a" yang spesifik untuk Bahasa Inggris.
    """
    def __init__(self):
        super().__init__()
        # Regex
        self.fraction_regex = re.compile(r"(((?:-?\d* )?-?\d+ *\/ *-? *\d+)|(-?\d* *(?:½|⅓|⅔|¼|¾|⅕|⅖|⅗|⅘|⅙|⅚|⅐|⅛|⅜|⅝|⅞|⅑|⅒)))")
        self.of_a_regex = re.compile(r"(-?\d+ -?\d+ *\/ *-? *\d+)|(-?\d+ *(?:½|⅓|⅔|¼|¾|⅕|⅖|⅗|⅘|⅙|⅚|⅐|⅛|⅜|⅝|⅞|⅑|⅒))")
        
        self.value_regex = re.compile(
            r"(-?(?: |\d)*\.?\d+ *(?:ribu|juta|miliar|triliun|kuadriliun|kuintiliun|sekstiliun|septiliun|oktiliun|undesiliun|tredesiliun|kuatuordesiliun|kuindesiliun|seksdesiliun|septendesiliun|oktodesiliun|novemdesiliun|vigintiliun)?)",
            flags=re.I
        )
        
        # Regex filter 
        self.filter_regex = re.compile(r"[,]")
        self.filter_space_regex = re.compile(r"[ ]")
        self.letter_filter_regex = re.compile(r"[^0-9\-\.]")

        # Dict prefix
        self.prefix_dict = {
            "Y": "yotta", "Z": "zetta", "E": "exa", "P": "peta", "T": "tera",
            "G": "giga", "M": "mega", "k": "kilo", "h": "hecto", "da": "deca",
            "d": "deci", "c": "centi", "m": "milli", "μ": "micro", "µ": "micro",
            "n": "nano", "p": "pico", "f": "femto", "a": "atto", "z": "zepto", "y": "yocto"
        }

        # Plural dan singular disamakan untuk Bahasa Indonesia.
        self.prefixable_trans_dict = {
            "m": {"singular": "meter", "plural": "meter"},
            "b": {"singular": "bit", "plural": "bit"},
            "B": {"singular": "bita", "plural": "bita"},
            "bps": {"singular": "bit per detik", "plural": "bit per detik"},
            "Bps": {"singular": "bita per detik", "plural": "bita per detik"},
            "g": {"singular": "gram", "plural": "gram"},
            "gf": {"singular": "gaya gram", "plural": "gaya gram"},
            "W": {"singular": "watt", "plural": "watt"},
            "Wh": {"singular": "watt jam", "plural": "watt jam"},
            "Hz": {"singular": "hertz", "plural": "hertz"},
            "hz": {"singular": "hertz", "plural": "hertz"},
            "J": {"singular": "joule", "plural": "joule"},
            "L": {"singular": "liter", "plural": "liter"},
            "V": {"singular": "volt", "plural": "volt"},
            "f": {"singular": "farad", "plural": "farad"},
            "s": {"singular": "detik", "plural": "detik"},
            "A": {"singular": "ampere", "plural": "ampere"},
            "Ah": {"singular": "ampere jam", "plural": "ampere jam"},
            "Pa": {"singular": "pascal", "plural": "pascal"},
            "C": {"singular": "coulomb", "plural": "coulomb"},
            "Bq": {"singular": "becquerel", "plural": "becquerel"},
            "N": {"singular": "newton", "plural": "newton"},
            "bar": {"singular": "bar", "plural": "bar"},
            "lm": {"singular": "lumen", "plural": "lumen"},
            "cal": {"singular": "kalori", "plural": "kalori"},
        }

        # Dict prefix
        self.prefixed_dict = {prefix + prefixed: {"singular": self.prefix_dict[prefix] + self.prefixable_trans_dict[prefixed]["singular"], "plural": self.prefix_dict[prefix] + self.prefixable_trans_dict[prefixed]["plural"]} for prefixed in self.prefixable_trans_dict for prefix in self.prefix_dict}
        self.prefixed_dict = {**self.prefixed_dict, **self.prefixable_trans_dict}

        # Dict unit kustom
        self.custom_dict = {
            "%": {"singular": "persen", "plural": "persen"},
            "pc": {"singular": "persen", "plural": "persen"},
            "ft": {"singular": "kaki", "plural": "kaki"},
            "mi": {"singular": "mil", "plural": "mil"},
            "mb": {"singular": "megabita", "plural": "megabita"},
            "ha": {"singular": "hektar", "plural": "hektar"},
            "\"": {"singular": "inci", "plural": "inci"},
            "in": {"singular": "inci", "plural": "inci"},
            "\'": {"singular": "kaki", "plural": "kaki"},
            "rpm": {"singular": "putaran per menit", "plural": "putaran per menit"},
            "hp": {"singular": "daya kuda", "plural": "daya kuda"},
            "cc": {"singular": "c c", "plural": "c c"},
            "oz": {"singular": "ons", "plural": "ons"},
            "mph": {"singular": "mil per jam", "plural": "mil per jam"},
            "lb": {"singular": "pon", "plural": "pon"},
            "lbs": {"singular": "pon", "plural": "pon"},
            "kt": {"singular": "knot", "plural": "knot"},
            "dB": {"singular": "desibel", "plural": "desibel"},
            "AU": {"singular": "satuan astronomi", "plural": "satuan astronomi"},
            "st": {"singular": "stone", "plural": "stone"},
            "yd": {"singular": "yard", "plural": "yard"},
            "yr": {"singular": "tahun", "plural": "tahun"},
            "yrs": {"singular": "tahun", "plural": "tahun"},
            "eV": {"singular": "volt elektron", "plural": "volt elektron"},
            "/": {"singular": "per", "plural": "per"},
            "sq": {"singular": "persegi", "plural": "persegi"},
            "2": {"singular": "persegi", "plural": "persegi"},
            "²": {"singular": "persegi", "plural": "persegi"},
            "3": {"singular": "kubik", "plural": "kubik"},
            "³": {"singular": "kubik", "plural": "kubik"},
            "h": {"singular": "jam", "plural": "jam"},
            "hr": {"singular": "jam", "plural": "jam"},
            "hrs": {"singular": "jam", "plural": "jam"},
            "ch": {"singular": "rantai", "plural": "rantai"},
            "KiB": {"singular": "kibibita", "plural": "kibibita"},
            "MiB": {"singular": "mebibita", "plural": "mebibita"},
            "GiB": {"singular": "gibibita", "plural": "gibibita"},
            "pH": {"singular": "p h", "plural": "p h"},
            "kph": {"singular": "kilometer per jam", "plural": "kilometer per jam"},
            "Da": {"singular": "dalton", "plural": "dalton"},
            "cwt": {"singular": "hundredweight", "plural": "hundredweight"},
            "Sv": {"singular": "sievert", "plural": "sievert"},
            "C": {"singular": "celcius", "plural": "celcius"}, # Menimpa Coulomb
            "degrees": {"singular": "derajat", "plural": "derajat"},
            "degree": {"singular": "derajat", "plural": "derajat"},
            "atm": {"singular": "atmosfer", "plural": "atmosfer"},
            "min": {"singular": "menit", "plural": "menit"},
            "cd": {"singular": "kandela", "plural": "kandela"},
            "ly": {"singular": "tahun cahaya", "plural": "tahun cahaya"},
            "kts": {"singular": "knot", "plural": "knot"},
            "mol": {"singular": "mol", "plural": "mol"},
            "Nm": {"singular": "newton meter", "plural": "newton meter"},
            "Ω": {"singular": "ohm", "plural": "ohm"},
            "bbl": {"singular": "barel", "plural": "barel"},
            "gal": {"singular": "galon", "plural": "galon"},
            "cal": {"singular": "kaliber", "plural": "kaliber"}, # Menimpa kalori, sesuai komentar asli
        }
        
        # Gabungkan dict
        self.prefixed_dict = {**self.prefixed_dict, **self.custom_dict}
        self.lower_prefixed_dict = {key.lower(): self.prefixed_dict[key] for key in self.prefixed_dict}

        # Regex pemisah ("per", "sq", "2", "3" akan diterjemahkan oleh dict)
        self.special_suffixes = re.compile(r"(\/|per(?!cent)|sq|2|²|3|³)")

        self.decimal = Decimal()
        self.fraction = Fraction()

    def convert(self, token: str) -> str:
        # 1 Filter koma
        token = self.filter_regex.sub("", token)
        result_list = []
        plural = False

        # 2 Coba match pecahan
        match = self.fraction_regex.match(token)
        if match:
            # 2.1 Konversi menggunakan Fraction
            result_list.append(self.fraction.convert(match.group(0)))
            token = token[:match.span()[0]] + token[match.span()[1]:]
            token = self.filter_space_regex.sub("", token)

            # Jika ini pecahan campuran (misal "8 1/2"), set plural
            if self.of_a_regex.match(match.group(0)):
                plural = True
            # else:
            #     result_list.append("of an" if token and token[0] in list("aeiou") else "of a")
            
        else:
            # 3 Coba match desimal/angka
            match = self.value_regex.match(token)
            if match:
                # 3.1 Konversi menggunakan Decimal
                result_list.append(self.decimal.convert(self.filter_space_regex.sub("", match.group(1))))
                token = token[:match.span()[0]] + token[match.span()[1]:]
                
                # Logika plural (tidak berdampak banyak di B. Indonesia
                # karena singular == plural, tapi tetap jaga)
                if abs(float(self.letter_filter_regex.sub("", match.group(1)))) != 1 or "." in match.group(1):
                    plural = True

        per = False
        # 4 Iterasi sisa token (unit)
        for split_token in token.split(" "):
            for i, token_chunk in enumerate(self.split_token(split_token)):
                # Cari di dict (case sensitive)
                if token_chunk in self.prefixed_dict:
                    result_list.append(self.prefixed_dict[token_chunk]["plural" if plural and not per else "singular"])
                # Cari di dict (case insensitive)
                elif token_chunk.lower() in self.lower_prefixed_dict:
                    result_list.append(self.lower_prefixed_dict[token_chunk.lower()]["plural" if plural and not per else "singular"])
                else:
                    result_list.append(token_chunk)
                
                # Logika "per" 
                if result_list[-1] == "per" and i != 0:
                    per = True
                elif result_list[-1] not in ("persegi", "kubik"): # "square"/"cubic" -> "persegi"/"kubik"
                    per = False
        
        result = " ".join(result_list)

        # 5 Handle edge case: (dihapus karena "sentimeter kubik" lebih benar)
        # result = re.sub(r"cubic centimeters?", "c c", result)

        return result
    
    def split_token(self, token: str) -> str:
        # Fungsi ini mendeteksi simbol (/, 2, 3) atau kata ("per", "sq")
        # yang kemudian akan diterjemahkan oleh dict.
        while True:
            match = self.special_suffixes.search(token)
            if match:
                s1, s2 = match.span()
                if match.group(1) in ("sq", "2", "²", "3", "³"):
                    yield token[s1:s2]
                    if token[:s1]:
                        yield token[:s1]
                else:
                    if token[:s1]:
                        yield token[:s1]
                    yield token[s1:s2]
                
                token = token[s2:]
            else:
                if token:
                    yield token
                break