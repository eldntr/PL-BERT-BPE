from singleton_decorator import singleton
import re
from .Digit import Digit
from .Cardinal import Cardinal

@singleton
class Decimal:
    """
    Class ini mengkonversi angka desimal (termasuk notasi ilmiah)
    ke dalam bentuk terucap Bahasa Indonesia.
    
    Diasumsikan input menggunakan format standar (',' sebagai ribuan, '.' sebagai desimal).
    
    Contoh:
    "1,000.5" -> "seribu koma lima"
    "3.14" -> "tiga koma satu empat"
    "3.0" -> "tiga koma nol"
    "3.66E-49" -> "tiga koma enam enam kali sepuluh pangkat minus empat puluh sembilan"
    """
    def __init__(self):
        super().__init__()
        # Regex untuk deteksi "x.y" atau ".y" 
        self.decimal_regex = re.compile(r"(-?\d*)\.(\d+)(.*)")
        # Regex untuk deteksi angka 
        self.number_regex = re.compile(r"(-?\d+)(.*)")
        # Regex filter untuk menghapus koma ribuan
        self.filter_regex = re.compile(r"[,]")
        
        # Konverter Digit and Cardinal
        self.cardinal = Cardinal()
        self.digit = Digit()
        
        # Daftar sufiks yang mungkin
        self.suffixes = [
            "ribu", 
            "juta", 
            "miliar", 
            "triliun", 
            "kuadriliun", 
        ]
        
        # Regex untuk deteksi sufiks (di-update otomatis dari list di atas)
        self.suffix_regex = re.compile(f" *({'|'.join(self.suffixes)})", flags=re.I)
        
        # Regex untuk notasi ilmiah xEy (tidak diubah)
        self.e_suffix_regex = re.compile(r" *E(-?\d+)", flags=re.I)
    
    def convert(self, token: str) -> str:

        def normalize_separators(raw_token: str) -> str:
            match = re.match(r"(-?[\d.,]+)", raw_token)
            if not match:
                return raw_token
            number_part = match.group(1)
            rest = raw_token[match.end():]

            last_dot = number_part.rfind(".")
            last_comma = number_part.rfind(",")

            if "," in number_part and "." in number_part:
                if last_dot > last_comma:
                    number_part = number_part.replace(",", "")
                else:
                    number_part = number_part.replace(".", "")
                    number_part = number_part.replace(",", ".", 1)
            elif "," in number_part:
                if number_part.count(",") == 1:
                    number_part = number_part.replace(",", ".", 1)
                else:
                    number_part = number_part.replace(",", "")
            elif "." in number_part:
                segments = number_part.split(".")
                if len(segments) > 1 and all(len(segment) == 3 for segment in segments[1:] if segment != ""):
                    number_part = "".join(segments)
                else:
                    number_part = number_part.replace(",", "")
            else:
                number_part = number_part.replace(",", "")

            return number_part + rest

        token = normalize_separators(token)

        # 1 Filter koma (ribuan)
        token = self.filter_regex.sub("", token)

        number = ""
        decimal = ""

        # 2 Cek format x.y
        match = self.decimal_regex.match(token)
        if match:
            number = match.group(1)
            decimal = match.group(2)
            token = match.group(3)
        else:
            # 3 Cek format x
            match = self.number_regex.match(token)
            if match:
                number = match.group(1)
                token = match.group(2)

        # 4 Cocokkan sufiks (misal "juta")
        match = self.suffix_regex.match(token)
        suffix = ""
        if match:
            suffix = match.group(1).lower() # Ambil sufiks yang cocok
        else:
            # 5 Jika tidak, cocokkan notasi ilmiah xEy
            match = self.e_suffix_regex.match(token)
            if match:
                # "times ten to the y" -> "kali sepuluh pangkat y"
                # `self.cardinal.convert` akan menangani angka negatif (misal "minus empat puluh sembilan")
                suffix = f"kali sepuluh pangkat {self.cardinal.convert(match.group(1))}"

        result_list = []
        
        # 6, 7, 8 Logika untuk bagian desimal
        if len(decimal) > 0:
            if suffix == "" and number and decimal.isdigit():
                if set(decimal) == {"0"}:
                    if len(decimal) % 3 == 0:
                        number += decimal
                    decimal = ""
                elif len(decimal) % 3 == 0:
                    number += decimal
                    decimal = ""

        if len(decimal) > 0:
            result_list.append("koma")
            
            # Kasus spesial untuk ".0" (misal "3.0")
            if decimal == "0" and len(number) > 0 and len(suffix) == 0:
                result_list.append("nol")
            else:
                # Jika tidak, gunakan konversi Digit (misal "14" -> "satu empat")
                # Ini akan menggunakan class Digit Anda yang sudah dilokalkan
                # "05" -> "kosong lima"
                result_list.append(self.digit.convert(decimal))

        # 9 Tambahkan bagian angka di depan (jika ada)
        if number:
            result_list.insert(0, self.cardinal.convert(number))

        # 10 Tambahkan sufiks jika ada
        if suffix:
            result_list.append(suffix)

        result = " ".join(result_list)
        
        return result