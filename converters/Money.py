from singleton_decorator import singleton
import re, os
from .Cardinal import Cardinal
from .Digit import Digit

@singleton
class Money:
    """
    Versi Bahasa Indonesia dari converter 'Money'.
    
    Contoh:
    "Rp1.500" -> "seribu lima ratus rupiah"
    "$1.56" -> "satu dolar dan lima puluh enam sen"
    "Rp1.5 juta" -> "satu koma lima juta rupiah"
    "€3.5 million" -> "tiga koma lima juta euro"
    """
    def __init__(self):
        super().__init__()
        # Regex
        self.decimal_regex = re.compile(r"(.*?)(-?\d*)\.(\d+)(.*)")
        self.number_regex = re.compile(r"(.*?)(-?\d+)(.*)")
        self.filter_regex = re.compile(r"[, ]")

        # Dict mata uang utama 
        self.currencies = {
            "$": {
                "number": {"singular": "dolar", "plural": "dolar"},
                "decimal": {"singular": "sen", "plural": "sen"}
            },
            "usd": {
                "number": {"singular": "dolar amerika serikat", "plural": "dolar amerika serikat"},
                "decimal": {"singular": "sen", "plural": "sen"}
            },
            "€": {
                "number": {"singular": "euro", "plural": "euro"},
                "decimal": {"singular": "sen", "plural": "sen"}
            },
            "£": {
                "number": {"singular": "pon", "plural": "pon"},
                "decimal": {"singular": "peni", "plural": "peni"}
            },
            
            "idr": {
                "singular": "rupiah",
                "plural": "rupiah"
            },
            "rp": {
                "singular": "rupiah",
                "plural": "rupiah"
            },
            "rp.": {
                "singular": "rupiah",
                "plural": "rupiah"
            }
        }

        # Muat money.json, tapi timpa dengan definisi lokal di atas
        try:
            with open(os.path.join(os.path.dirname(__file__), "money.json"), "r") as f:
                import json
                self.currencies = {**json.load(f), **self.currencies}
        except FileNotFoundError:
            # Jika money.json tidak ada, gunakan saja dict di atas
            pass

        # List sufiks skala
        self.suffixes = [
            "ribu", 
            "juta", 
            "miliar", 
            "triliun", 
            "kuadriliun", 
            "kuintiliun", 
            "sekstiliun", 
            "septiliun", 
            "oktiliun", 
            "undesiliun", 
            "tredesiliun", 
            "kuatuordesiliun", 
            "kuindesiliun", 
            "seksdesiliun", 
            "septendesiliun", 
            "oktodesiliun", 
            "novemdesiliun", 
            "vigintiliun"
        ]

        # Dict singkatan sufiks 
        self.abbr_suffixes = {
            "k": "ribu",
            "m": "juta",
            "bn": "miliar",
            "b": "miliar",
            "t": "triliun",
        }

        # Regex untuk deteksi sufiks
        self.suffix_regex = re.compile(r"(kuatuordesiliun|septendesiliun|novemdesiliun|kuindesiliun|oktodesiliun|seksdesiliun|tredesiliun|kuadriliun|kuintiliun|undesiliun|sekstiliun|septiliun|vigintiliun|oktiliun|triliun|miliar|juta|ribu|bn|k|m|b|t)(.*)", flags=re.I)
        
        # Regex untuk deteksi mata uang (ditambahkan 'rp' dan 'rp\.')
        self.currency_regex = re.compile(r"(.*?)(dollar|usd|rs\.|r\$|aed|afn|all|amd|ang|aoa|ars|aud|awg|azn|bam|bbd|bdt|bgn|bhd|bif|bmd|bnd|bob|brl|bsd|btc|btn|bwp|byn|bzd|cad|cdf|chf|clf|clp|cnh|cny|cop|crc|cuc|cup|cve|czk|djf|dkk|dop|dzd|egp|ern|etb|eur|fjd|fkp|gbp|gel|ggp|ghs|gip|gmd|gnf|gtq|gyd|hkd|hnl|hrk|htg|huf|idr|ils|imp|inr|iqd|irr|isk|jep|jmd|jod|jpy|kes|kgs|khr|kmf|kpw|krw|kwd|kyd|kzt|lak|lbp|lkr|lrd|lsl|lyd|mad|mdl|mga|mkd|mmk|mnt|mop|mro|mru|mur|mvr|mwk|mxn|myr|mzn|nad|ngn|nio|nok|npr|nzd|omr|pab|pen|pgk|php|pk|pln|pyg|qar|ron|rsd|rub|rwf|sar|sbd|scr|sdg|sek|sgd|shp|sll|sos|srd|ssp|std|stn|svc|syp|szl|thb|tjs|tmt|tnd|top|try|ttd|twd|tzs|uah|ugx|usd|uyu|uzs|vef|vnd|vuv|wst|xaf|xag|xau|xcd|xdr|xof|xpd|xpf|xpt|yer|zar|zmw|zwl|fim|bef|cyp|ats|ltl|zl|u\$s|rp|rp\.|rs|tk|r$|dm|\$|€|£|¥)(.*?)", flags=re.I)

        self.cardinal = Cardinal()
        self.digit = Digit()

    def convert(self, token: str) -> str:

        # 1 Hapus koma dan spasi
        token = self.filter_regex.sub("", token)

        before = ""
        after = ""
        currency = None
        number = ""
        decimal = ""
        scale = ""

        # 2 Coba match desimal "x.y"
        match = self.decimal_regex.search(token[::-1])
        if match:
            before = match.group(4)[::-1]
            number = match.group(3)[::-1]
            decimal = match.group(2)[::-1]
            after = match.group(1)[::-1]
        
        else:
            # 3 Coba match integer
            match = self.number_regex.search(token)
            if match:
                before = match.group(1)
                number = match.group(2)
                after = match.group(3)
        
        # 4 Cek 'before' untuk mata uang
        if before:
            before_lower = before.lower()
            if before_lower in self.currencies:
                currency = self.currencies[before_lower]
            elif before_lower.endswith('.'): # Cek "Rp."
                if before_lower in self.currencies:
                     currency = self.currencies[before_lower]
                elif before_lower[:-1] in self.currencies: # Cek "rs."
                     currency = self.currencies[before_lower[:-1]]
            elif before[-1] in self.currencies:
                currency = self.currencies[before[-1]]

        # 5 Cek 'after' untuk mata uang dan sufiks
        if after:
            # 5.1 Cek sufiks skala (juta, miliar, dst.)
            match = self.suffix_regex.match(after)
            if match:
                scale_key = match.group(1).lower()
                scale = self.abbr_suffixes.get(scale_key, scale_key)
                after = match.group(2)

            # 5.2 Cek mata uang di 'after'
            if after.lower() in self.currencies:
                currency = self.currencies[after.lower()]
                after = ""
            elif after.lower().startswith('.') and after.lower()[1:] in self.currencies: # Cek ".rp"
                 currency = self.currencies[after.lower()[1:]]
                 after = ""


        # Cek apakah mata uang mendukung "sen" (misal Dolar)
        decimal_support = currency and "number" in currency and "decimal" in currency

        result_list = []
        if decimal_support and not scale:
            # 6 Logika "x dolar y sen"
            
            if number and (number != "0" or not decimal):
                result_list.append(self.cardinal.convert(number))
                result_list.append(currency["number"]["singular" if number == "1" else "plural"])
                if decimal and decimal != "0" * len(decimal):
                    result_list.append("dan") # "and" -> "dan"
            
            if decimal and decimal != "0" * len(decimal):
                # Pad desimal ke 2 digit (misal "5" -> "50")
                decimal = f"{decimal:0<2}"
                result_list.append(self.cardinal.convert(decimal))
                result_list.append(currency["decimal"]["singular" if decimal == "01" else "plural"])
        
        else:
            # 7 Logika "x koma y [skala] [mata uang]" (misal: "satu koma lima juta rupiah")
            if number:
                result_list.append(self.cardinal.convert(number))
            if decimal and decimal != "0" * len(decimal):
                result_list.append("koma") # "point" -> "koma"
                result_list.append(self.digit.convert(decimal)) # "5" -> "lima"
            
            if scale:
                result_list.append(scale)
            
            if currency:
                # Ambil nama mata uang (singular/plural)
                currency_name_dict = currency.get("number", currency)
                
                if number == "1" and not decimal and not scale:
                    result_list.append(currency_name_dict["singular"])
                else:
                    result_list.append(currency_name_dict["plural"])
        
        # 8 Tambahkan sisa 'after'
        if after:
            result_list.append(after.lower())

        result = " ".join(result_list)
        return result