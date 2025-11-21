from singleton_decorator import singleton
import re

@singleton
class Digit:
    """
    Class ini mengkonversi string digit (misal: "123", "081")
    ke dalam bentuk terucap per digit dalam Bahasa Indonesia.
    
    Contoh:
    "123" -> "satu dua tiga"
    "007" -> "kosong kosong tujuh"
    "081" -> "kosong delapan satu"
    """
    def __init__(self):
        super().__init__()
        # Regex untuk filter non-digit
        self.filter_regex = re.compile("[^0-9]")
        
        # Dict digit ke teks 
        self.trans_dict = {
            "0": "kosong",
            "1": "satu",
            "2": "dua",
            "3": "tiga",
            "4": "empat",
            "5": "lima",
            "6": "enam",
            "7": "tujuh",
            "8": "delapan",
            "9": "sembilan"
        }

    def convert(self, token: str) -> str:
        # 1 Filter semua yang bukan digit
        token = self.filter_regex.sub("", token)

        # 2 & 3 Konversi tiap digit ke teks dan gabungkan dengan spasi
        #   Misalkan "007" -> "kosong kosong tujuh"
        token = " ".join([self.trans_dict[c] for c in token])
        return token