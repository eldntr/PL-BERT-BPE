from singleton_decorator import singleton
import re
from .Verbatim import Verbatim

@singleton
class Letters:
    """
    Versi Bahasa Indonesia dari converter 'Letters'.
    Tugasnya adalah mengeja akronim atau huruf. (Misal: "ABC" -> "a b c")

    Perubahan dari aslinya:
    - Menghapus semua logika sufiks ('s, s') karena tidak relevan
      untuk plural/posesif Bahasa Indonesia.
    - Menghapus kamus terjemahan aksen (trans_dict).
    """
    def __init__(self):
        super().__init__()
        # Filter hanya huruf, &, dan '
        self.filter_regex = re.compile(r"[^A-Za-z&']")
        
        # Konversi Verbatim (bergantung pada Verbatim.py yang dilokalkan)
        self.verbatim = Verbatim()
        
        # Dict aksen dikosongkan
        self.trans_dict = {}
    
    def convert(self, token: str) -> str:
        
        # 1 Kasus "nan" (float) -> "n a"
        if type(token) == float:
            return "n a"
        
        # 2 Ambil kata pertama saja
        if " " in token and ". " not in token:
            token = token.split(" ")[0]

        # 3 Jika panjang token 1, konversi langsung
        if len(token) == 1:
            return self.convert_char(token)
        
        # 4 & 5: Logika Sufix ('s) Dihapus
        
        # Bersihkan token dari karakter non-alfabet
        token = self.filter_regex.sub("", str(token))

        # 6 Eja token, abaikan apostrof, tanpa menambahkan sufiks
        result_list = [
            self.convert_char(char) 
            for char in token 
            if char != "'"
        ]
        return " ".join(result_list)

    def convert_char(self, char: str) -> str:
        # self.trans_dict kosong di versi Indonesia,
        # jadi ini akan selalu memanggil Verbatim
        if char in self.trans_dict:
            return self.trans_dict[char]

        # Langsung panggil Verbatim.py untuk konversi per karakter
        # (misal: 'A' -> 'a', atau 'A' -> 'a' tergantung Verbatim.py)
        return self.verbatim.convert_char(char)