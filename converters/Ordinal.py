from singleton_decorator import singleton
import re
from .Roman import Roman
from .Cardinal import Cardinal

@singleton
class Ordinal:
    """
    Class ini mengkonversi angka ordinal (misal: "5th", "II", "ke-2")
    ke dalam bentuk terucap Bahasa Indonesia.
    
    Logika Bahasa Indonesia:
    - 1 -> "pertama"
    - 2 -> "kedua"
    - 22 -> "kedua puluh dua"
    - II -> "kedua"
    """
    def __init__(self):
        super().__init__()
        # Regex untuk filter koma, spasi, dan indikator ordinal (ºª)
        self.filter_regex = re.compile(r"[, ºª]")
        
        # Regex untuk mendeteksi format Inggris (misal "5th", "22nd")
        self.standard_case_regex = re.compile(r"(?i)(\d+)(th|nd|st|rd)(s?)")
        
        # Regex untuk mendeteksi format Indonesia (misal "ke-5", "ke5")
        self.indo_prefix_regex = re.compile(r"(?i)(ke-?)(\d+)")

        self.roman = Roman()
        self.cardinal = Cardinal()

        # Dict 'self.trans_denominator' (versi Inggris) dihapus
        # karena tidak diperlukan untuk logika 'ke-'.
    
    def convert(self, token: str) -> str:
        
        # 1 Filter koma, spasi, ºª
        token = self.filter_regex.sub("", token)

        # Logika prefiks/sufiks 'the' dan 's (versi Inggris) dihapus.
        
        number_str = None

        # 2 Cek apakah token adalah Angka Romawi (misal "II")
        if self.roman.check_if_roman(token):
            # localized roman.convert() akan mengembalikan ("2", "")
            number_str, _ = self.roman.convert(token)
        
        else:
            # 3 Cek format Inggris (misal "5th")
            match = self.standard_case_regex.fullmatch(token)
            if match:
                number_str = match.group(1)
            else:
                # 4 Cek format Indonesia (misal "ke-5")
                match = self.indo_prefix_regex.fullmatch(token)
                if match:
                    number_str = match.group(2)

        # Jika tidak ada format di atas yang cocok, asumsikan token
        # adalah angka biasa (misal "5" yang diklasifikasikan sebagai ORDINAL)
        if number_str is None:
            number_str = token

        try:
            # 5.A Kasus Spesial untuk "1"
            if number_str == "1":
                return "pertama"
            
            # 5.B Kasus Umum
            # Konversi angka (misal "22") ke kardinal ("dua puluh dua")
            cardinal_text = self.cardinal.convert(number_str)
            
            # Tambahkan awalan "ke-"
            # "ke" + "dua puluh dua" -> "kedua puluh dua"
            return f"ke{cardinal_text}"
            
        except Exception:
            # Fallback jika input tidak valid (misal "abc")
            return token