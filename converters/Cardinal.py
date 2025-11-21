from singleton_decorator import singleton
import re
from .Roman import Roman  # Asumsi Roman.py tidak perlu diubah

@singleton
class Cardinal:
    """
    Class ini mengkonversi angka kardinal (misal: "123", "-10", "IV")
    ke dalam format ucapan Bahasa Indonesia.
    
    Langkah-langkah:
    - 1 Hapus titik.
    - 2 Cek apakah ini Angka Romawi.
    - 3 ... (Logika dari kode asli)
    
    Kasus Spesial (Indonesia):
    10 -> sepuluh
    11 -> sebelas
    100 -> seratus
    123 -> seratus dua puluh tiga
    1000 -> seribu
    1100 -> seribu seratus
    2000 -> dua ribu
    -2 -> minus two
    """
    def __init__(self):
        super().__init__()
        # Regex untuk filter non-digit (kecuali "-")
        self.filter_regex = re.compile("[^0-9\-]")
        self.filter_strict_regex = re.compile("[^0-9]")
        self.dot_filter_regex = re.compile("[.]")

        # Daftar sufiks skala
        self.scale_suffixes = [
            "ribu", 
            "juta", 
            "miliar", 
            "triliun", 
            "kuadriliun",
        ]

        # Dict angka kecil
        self.small_trans_dict = {
            "1": "satu",
            "2": "dua",
            "3": "tiga",
            "4": "empat",
            "5": "lima",
            "6": "enam",
            "7": "tujuh",
            "8": "delapan",
            "9": "sembilan",
        }

        # Dict puluhan
        self.tens_trans_dict = {
            "1": "sepuluh", 
            "2": "dua puluh",
            "3": "tiga puluh",
            "4": "empat puluh",
            "5": "lima puluh",
            "6": "enam puluh",
            "7": "tujuh puluh",
            "8": "delapan puluh",
            "9": "sembilan puluh",
        }

        # Dict kasus spesial
        self.special_trans_dict = {
            11: "sebelas",
            12: "dua belas",
            13: "tiga belas",
            14: "empat belas",
            15: "lima belas",
            16: "enam belas",
            17: "tujuh belas",
            18: "delapan belas",
            19: "sembilan belas"
        }

        # Konversi Romawi
        self.roman = Roman()

    def _give_chunk(self, num_str: str, size:int = 3) -> str:
        # Memecah string angka menjadi chunk dari belakang
        while num_str:
            yield num_str[-size:]
            num_str = num_str[:-size]

    def convert(self, token: str) -> str:
        # 1 Hapus Titik
        token = self.dot_filter_regex.sub("", token)

        suffix = ""
        # 2 Cek Angka Romawi
        if self.roman.check_if_roman(token):
            token, suffix = self.roman.convert(token)

        # 3 Filter non-digit (tetap simpan "-")
        token = self.filter_regex.sub("", token)

        # 5 Cek prefix "minus"
        prefix = ""
        while len(token) > 0 and token[0] == "-":
            token = token[1:]
            prefix = "minus" if prefix == "" else "" # "minus" tetap dipakai

        # 6 Hapus sisa "-"
        token = self.filter_strict_regex.sub("", token)

        text_list = []

        # 7 Kasus "0"
        if token == len(token) * "0":
            text_list.append("nol")
        else:
            # 8 Pecah angka menjadi chunk
            for depth, chunk in enumerate(self._give_chunk(token)):
                chunk_text_list = []
                # 9 Pecah chunk menjadi ratusan dan sisa
                hundred, rest = chunk[-3:-2], chunk[-2:]
                
                # 10 Dapatkan "x ratus"
                if len(hundred) != 0 and int(hundred) != 0:
                    if hundred == "1":
                        chunk_text_list.append("seratus")
                    else:
                        chunk_text_list.append(self.small_trans_dict[hundred])
                        chunk_text_list.append("ratus")
                
                # 11 Dapatkan teks dari `rest` (puluhan dan satuan)
                if int(rest) in self.special_trans_dict:
                    chunk_text_list.append(self.special_trans_dict[int(rest)])
                else:
                    if len(rest) == 2 and rest[-2] != "0":
                        # Ambil dari 'tens_trans_dict' ("sepuluh", "dua puluh", dst.)
                        chunk_text_list.append(self.tens_trans_dict[rest[-2]])
                    
                    if rest[-1] != "0":
                        # Jika 10, 'tens_trans_dict' sudah "sepuluh", rest[-1] '0' di-skip.
                        # Jika 21, 'tens_trans_dict' "dua puluh", 'small_trans_dict' "satu".
                        chunk_text_list.append(self.small_trans_dict[rest[-1]])
                
                # 12 Tambah sufiks skala (ribu, juta, dst.)
                if depth > 0 and len(chunk_text_list) > 0:
                    try:
                        scale_word = self.scale_suffixes[depth-1]
                        
                        # Jika chunk-nya ["satu"] dan skala-nya "ribu" -> ganti jadi "seribu"
                        if scale_word == "ribu" and chunk_text_list == ["satu"]:
                            chunk_text_list = ["seribu"]
                        else:
                            chunk_text_list.append(scale_word)
                            
                    except IndexError:
                        # Angka terlalu besar, tidak ada sufiks
                        pass
                
                # 13 Gabungkan hasil chunk ke list utama
                text_list = chunk_text_list + text_list
        
        # 14 Gabungkan list menjadi string
        token = " ".join(text_list)

        # 15 Terapkan pre/sufiks
        if prefix:
            token = f"{prefix} {token}"
        if suffix:
            token = f"{token}{suffix}"

        return token