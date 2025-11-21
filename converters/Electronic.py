from singleton_decorator import singleton
import re
from .Cardinal import Cardinal
from .Digit import Digit

@singleton
class Electronic:
    """
    Class ini mengkonversi token elektronik (URL, email, hashtag)
    ke dalam bentuk terucap Bahasa Indonesia.
    
    Contoh:
    "#Text" -> "tanda pagar text"
    "website.com" -> "website dot com"
    "nama@email.id" -> "nama et email dot i d"
    """
    def __init__(self):
        super().__init__()
        
        # --- Dict untuk 'data_convert' (metode 1) ---
        
        # Dict untuk URL yang diawali http(s)
        self.data_https_dict = {
            "/": "garis miring",
            ":": "titik dua",
            ".": "dot",  
            "#": "pagar",
            "-": "strip",
            "@": "et",

            "(": "k u r u n g b u k a",
            ")": "k u r u n g t u t u p",
            "_": "g a r i s b a w a h",
            ",": "k o m a",
            "%": "p e r s e n",
            "~": "t i l d e",
            ";": "t i t i k k o m a",
            "'": "k u t i p t u n g g a l",
            "\"": "k u t i p g a n d a",

            "0": "kosong",
            "1": "s a t u",
            "2": "d u a",
            "3": "t i g a",
            "4": "e m p a t",
            "5": "l i m a",
            "6": "e n a m",
            "7": "t u j u h",
            "8": "d e l a p a n",
            "9": "s e m b i l a n",
        }

        # Dict untuk token yang TIDAK diawali http(s)
        self.data_no_https_dict = {
            "/": "g a r i s m i r i n g",
            ":": "t i t i k d u a",
            ".": "dot",  # <-- Sudah "dot"
            "#": "p a g a r",
            "-": "s t r i p",
            "@": "e t",

            "(": "k u r u n g b u k a",
            ")": "k u r u n g t u t u p",
            "_": "g a r i s b a w a h",
            ",": "k o m a",
            "%": "p e r s e n",
            "~": "t i l d e",
            ";": "t i t i k k o m a",
            "'": "k u t i p t u n g g a l",
            "\"": "k u t i p g a n d a",
            
            "0": "kosong",
            "1": "s a t u",
            "2": "d u a",
            "3": "t i g a",
            "4": "e m p a t",
            "5": "l i m a",
            "6": "e n a m",
            "7": "t u j u h",
            "8": "d e l a p a n",
            "9": "s e m b i l a n",
        }

        self.data_http_regex = re.compile(r"https?://")
        
        # --- Dict untuk 'sensible_convert' (metode 2) ---
        
        self.sensible_trans_dict = {
            "/": "garis miring",
            ":": "titik dua",
            ".": "dot",  
            "#": "tanda pagar",
            "-": "strip",
            "_": "garis bawah",
            ",": "koma",
            "%": "persen",
            "~": "tilde",
            ";": "titik koma",
            "'": "kutip tunggal",
            "\"": "kutip ganda",
            "@": "et", 

            "0": "kosong",
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

        # Menggunakan class Cardinal dan Digit yang sudah dilokalkan
        self.cardinal = Cardinal()
        self.digit = Digit()

    def convert(self, token: str) -> str:
        # 1 Ubah ke lowercase
        token = token.lower()

        # 2 Kasus spesial "::"
        if token == "::":
            return token

        # 3 Kasus spesial "#Text" -> "tanda pagar text"
        if token[0] == "#" and len(token) > 1:
            return self.convert_hash_tag(token)

        # Cek apakah diawali http(s)://
        http = self.data_http_regex.match(token) != None
        data_trans_dict = self.data_https_dict if http else self.data_no_https_dict
        
        result_list = []
        c_index = 0
        while c_index < len(token):
            if http:
                # 4.1 Jika .com, tambahkan "dot com"
                if token[c_index:].startswith(".com"):
                    result_list.append("dot com")
                    c_index += len(".com")
                    continue
            
            # 4.2 Cek angka
            offset = 0
            while c_index + offset < len(token) and token[c_index + offset].isdigit():
                offset += 1
            
            # Gunakan Cardinal atau Digit (yang sudah dilokalkan)
            if offset == 2 and token[c_index] != "0":
                text = self.cardinal.convert(token[c_index:c_index + offset])
                result_list.append(" ".join([c for c in text if c != " "]))
                c_index += offset
            elif offset > 0 and token[c_index] != "0" * offset:
                text = self.digit.convert(token[c_index:c_index + offset])
                result_list.append(" ".join([c for c in text if c != " "]))
                c_index += offset
            else:
                # 4.3 Tambahkan karakter non-numerik dari dict terjemahan
                if token[c_index] in data_trans_dict:
                    result_list.append(data_trans_dict[token[c_index]])
                else:
                    result_list.append(token[c_index]) 
                c_index += 1

        return " ".join(result_list)
    
    # Metode 'sensible'
    def sensible_convert(self, token: str) -> str:
        # 1 Ubah ke lowercase
        token = token.lower()

        # 2 Kasus spesial "::"
        if token == "::":
            return token

        # 3 Kasus spesial "#Text"
        if token[0] == "#" and len(token) > 1:
            return self.convert_hash_tag(token)

        result_list = []
        c_index = 0
        while c_index < len(token):            
            # 4.1 Cek domain .co.id
            if token[c_index:].startswith(".co.id"):
                result_list.append("dot c o dot i d")
                c_index += 6
                continue
            # 4.2 Cek domain .id
            elif token[c_index:].startswith(".id"):
                result_list.append("dot i d")
                c_index += 3
                continue
            # 4.3 Cek domain .com
            elif token[c_index:].startswith(".com"):
                result_list.append("dot com")
                c_index += 4
                continue
            
            # 4.4 Konversi karakter menggunakan dict 'sensible'
            char = token[c_index]
            if char in self.sensible_trans_dict:
                result_list.append(self.sensible_trans_dict[char])
            else:
                result_list.append(char)

            c_index += 1
        
        return " ".join(result_list)
    
    def convert_hash_tag(self, token: str) -> str:
        # "hash tag" -> "tanda pagar"
        out = "tanda pagar "
        for char in token[1:].lower():
            if char in self.sensible_trans_dict:
                if out[-1] == " ":
                    out += self.sensible_trans_dict[char] + " "
                else:
                    out += " " + self.sensible_trans_dict[char] + " "
            else:
                out += char
        return out.strip()