from singleton_decorator import singleton
import re

@singleton
class Telephone:
    """
    Versi Bahasa Indonesia dari converter 'Telephone'.
    Mengubah nomor telepon, ekstensi, dll. ke format terucap.
    
    Contoh (setelah lokalisasi):
    "081-500" -> "kosong delapan satu sil lima ratus"
    "x1000" -> "ekstensi satu ribu"
    "021" -> "kosong dua satu"
    """
    def __init__(self):
        super().__init__()
        # Kamus terjemahan
        self.trans_dict = {
            " ": "sil", # "sil" (silence) digunakan sebagai token internal untuk spasi/jeda
            "-": "sil",

            "x": "ekstensi", # "extension" -> "ekstensi"

            "0": "kosong",   # "o" -> "kosong"
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
        # Regex untuk filter kurung
        self.filter_regex = re.compile(r"[()]")

    def convert(self, token: str) -> str:
        # 1 Ubah ke lowercase dan ganti kurung dengan strip
        token = self.filter_regex.sub("-", token.lower())

        # 2 Konversi list karakter menggunakan dict terjemahan
        result_list = [self.trans_dict[c] if c in self.trans_dict else c for c in token]

        # 3 Hapus "sil" berurutan atau di awal
        result_list = [section for i, section in enumerate(result_list) if section != "sil" or (i - 1 >= 0 and result_list[i - 1] != "sil")]

        # 4 Iterasi dan ganti "kosong" berurutan dengan "ratus" atau "ribu"
        # (Logika disesuaikan untuk "kosong", "ratus", "ribu")
        i = 0
        while i < len(result_list):
            offset = 0
            while i + offset < len(result_list) and result_list[i + offset] == "kosong":
                offset += 1

            # Cek kondisi (sebelumnya harus angka, bukan "kosong" atau "sil")
            if (i + offset >= len(result_list) or result_list[i + offset] == "sil") and \
               (i - 1 < 0 or result_list[i - 1] not in ("kosong", "sil")) and \
               offset in (2, 3):
                
                # Ganti "kosong kosong" -> "ratus", "kosong kosong kosong" -> "ribu"
                result_list[i : offset + i] = ["ratus"] if offset == 2 else ["ribu"]

            i += 1

        return " ".join(result_list)