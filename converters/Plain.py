from singleton_decorator import singleton
import re, os

@singleton
class Plain:
    """
    Versi Bahasa Indonesia dari converter 'Plain'.
    Ini adalah converter "catch-all" untuk token yang tidak
    terklasifikasi sebagai tipe lain.
    
    Perubahan:
    - Mengganti kamus 'upper_trans_dict' dan 'trans_dict' dengan
      singkatan umum Bahasa Indonesia (misal: Jl, Kab, RT, RW).
    - Menghapus pemuatan 'plain.json' (spesifik B. Inggris).
    - Menghapus 'split_at' (spesifik B. Jerman).
    """
    def __init__(self):
        super().__init__()
        # Dict untuk token uppercase (Sensitif huruf besar)
        self.upper_trans_dict = {
            "JLN": "jalan",
            "JL": "jalan",
            "DS": "desa",
            "KEL": "kelurahan",
            "KEC": "kecamatan",
            "KAB": "kabupaten",
            "RT": "erte",
            "RW": "erwe",
            "NO": "nomor",
            "DR": "dokter",
            "H": "haji",
            "HJ": "hajah",
            "PT": "perseroan terbatas",
            "CV": "persekutuan komanditer"
        }

        # Dict untuk token lowercase
        self.trans_dict = {
            "jln": "jalan",
            "jl": "jalan",
            "ds": "desa",
            "kel": "kelurahan",
            "kec": "kecamatan",
            "kab": "kabupaten",
            "rt": "erte",
            "rw": "erwe",
            "no": "nomor",
            "dr": "dokter",
            "h": "haji",
            "hj": "hajah",
            "pt": "perseroan terbatas",
            "cv": "persekutuan komanditer"
        }

        # Dihapus karena plain.json adalah untuk konversi UK -> US
        # with open(os.path.join(os.path.dirname(__file__), "plain.json")) as f:
        #     import json
        #     self.trans_dict = {**self.trans_dict, **json.load(f)}

        # Dikosongkan karena "strasse" (Jerman) tidak relevan
        self.split_at = []

        # Regex ini sekarang aman karena self.split_at kosong
        self.split_at_regex = re.compile(f"(.*)({'|'.join(self.split_at)})$", flags=re.I)
    
    def convert(self, token: str) -> str:
        # 1 Kasus "NaN" (float)
        if isinstance(token, float):
            return "NaN"
        
        # 2 Cek kamus uppercase
        if token in self.upper_trans_dict:
            return self.upper_trans_dict[token]

        # 3 Cek kamus lowercase
        if token.lower() in self.trans_dict:
            return self.trans_dict[token.lower()]

        # 4 Hapus karakter non-alfanumerik
        # Regex ini dipertahankan karena juga menangani diakritik
        token = re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9']", "", token)

        # 5 Logika split (tidak akan berjalan karena self.split_at kosong)
        if token.lower().endswith(tuple(self.split_at)):
            groups = self.split_at_regex.match(token).groups()
            if groups[0]:
                token = " ".join(groups).lower()

        return token