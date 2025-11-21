from singleton_decorator import singleton
import re
from .Cardinal import Cardinal
from .Digit import Digit

@singleton
class Address:
    """
    Class ini mengkonversi token alamat (misal: "A-101", "Blok C05", "I02B")
    ke dalam format ucapan Bahasa Indonesia.
    
    Contoh:
    "A-101" -> "a satu kosong satu"
    "I02B"  -> "i kosong dua barat"
    """
    def __init__(self):
        super().__init__()
        # Regex untuk filter spasi, titik, dan dash (strip)
        self.filter_regex = re.compile(r"[. -]")
        
        # Regex untuk deteksi alamat.
        # Bagian suffix diubah dari [NESWnesw] menjadi [UTSButsb]
        self.address_regex = re.compile(
            r"((?P<upper_prefix>[A-Z\.]*)|(?P<lower_prefix>[a-zA-Z]*))"
            r"(?P<link>( |-)*)"
            r"(?P<number>\d+)"
            r"(?P<suffix>U|T|S|B|u|t|s|b)?"
        )

        # Dict untuk arah mata angin
        self.direction_trans_dict = {
            "u": "utara",   # U utk Utara (North)
            "t": "timur",   # T utk Timur (East)
            "s": "selatan", # S utk Selatan (South)
            "b": "barat"    # B utk Barat (West)
        }

        # Konverter Cardinal dan Digit
        self.cardinal = Cardinal()
        self.digit = Digit()

    def convert(self, token: str) -> str:
        
        # 1 Strip spasi dari token
        token = token.strip()
        
        result_list = []

        # 2 Coba match dengan regex alamat
        match = self.address_regex.match(token)
        if match:
            # Ekstrak nilai dari match
            lower_prefix, upper_prefix, link, number, suffix = (
                match.group("lower_prefix"),
                match.group("upper_prefix"),
                match.group("link"),
                match.group("number"),
                match.group("suffix")
            )

            # 2.1 Logika prefix
            # lower_prefix (misal "Jalan") ditambahkan sebagai lowercase.
            # upper_prefix (misal "JLN" atau "BLOK") dieja per huruf.
            if lower_prefix:
                result_list.append(lower_prefix.lower())
            elif upper_prefix:
                result_list += [c for c in upper_prefix.lower() if c != "."]

            # 2.2 Logika konversi angka
            # Kita menggunakan konversi parsial jika panjang angka 2, atau kadang 3
            if ((link or number[-1] == "0" or number[0] == "0") and len(number) == 3) or len(number) == 2:
                if number[-3:-2]:
                    result_list.append(self.digit.convert(number[-3:-2]))
                
                # Logika untuk "0" di tengah (misal "101" -> "satu KOSONG satu")
                if number[-2:-1] == "0":
                    result_list.append("kosong") 
                    result_list.append(self.digit.convert(number[-1]))
                else:
                    # Misal "25" -> "dua puluh lima" (dihandle oleh Cardinal)
                    result_list.append(self.cardinal.convert(number[-2:]))
            
            else:
                # Jika tidak, gunakan konversi digit (misal "123" -> "satu dua tiga")
                result_list.append(self.digit.convert(number))
            
            # 2.3 Jika ada suffix (mata angin), tambahkan terjemahannya
            if suffix:
                result_list.append(self.direction_trans_dict[suffix.lower()])

            return " ".join(result_list)
        
        # Jika tidak cocok regex, kembalikan token asli
        return token