from singleton_decorator import singleton
import re

@singleton
class Verbatim:
    """
    Class 'Verbatim' yang dilokalkan untuk Bahasa Indonesia.
    
    Tugasnya adalah mengkonversi token "verbatim", seringkali
    simbol individu atau mengeja akronim karakter per karakter.
    
    Contoh:
    "#" -> "tanda pagar"
    "&" -> "dan"
    "Rp" -> "rupiah"
    ".6-cM" -> "titik e n a m s t r i p c m"
    """
    def __init__(self):
        super().__init__()

        # Dict untuk token/karakter spesial
        self.trans_dict = {
            # Karakter
            "&": "dan",
            "_": "garis bawah",
            "#": "tanda pagar",
            "~": "tilde",
            "%": "persen",
            "@": "et",

            # Mata Uang
            "€": "euro",
            "$": "dolar",
            "£": "pon",
            "Rp": "rupiah",
            "rp": "rupiah",

            # Matematika
            "²": "kuadrat",
            "³": "kubik",
            "×": "kali",
            "=": "sama dengan",
            ">": "lebih besar dari",
            "<": "kurang dari",

            # Yunani
            "α": "alpha", "Α": "alpha",
            "β": "beta", "Β": "beta",
            "γ": "gamma", "Γ": "gamma",
            "δ": "delta", "Δ": "delta",
            "ε": "epsilon", "Ε": "epsilon",
            "ζ": "zeta", "Ζ": "zeta",
            "η": "eta", "Η": "eta",
            "θ": "theta", "Θ": "theta",
            "ι": "iota", "Ι": "iota",
            "κ": "kappa", "Κ": "kappa",
            "λ": "lambda", "Λ": "lambda",
            "Μ": "mu", "μ": "mu",
            "ν": "nu", "Ν": "nu",
            "ξ": "xi", "Ξ": "xi",
            "ο": "omicron", "Ο": "omicron",
            "π": "pi", "Π": "pi",
            "ρ": "rho", "Ρ": "rho",
            "ς": "sigma", "σ": "sigma", "Σ": "sigma", "Ϲ": "sigma", "ϲ": "sigma",
            "τ": "tau", "Τ": "tau",
            "υ": "upsilon", "Υ": "upsilon",
            "φ": "phi", "Φ": "phi",
            "χ": "chi", "Χ": "chi",
            "ψ": "psi", "Ψ": "psi",
            "ω": "omega", "Ω": "omega",

            # Pengukuran
            "µ": "mikro"
        }

        # Dict untuk mengeja karakter
        # Nama "trans_ordinal_dict" dari aslinya, tapi fungsinya untuk mengeja.
        self.trans_ordinal_dict = {
            ".": "titik",
            "-": "s t r i p", # "d a s h" -> "s t r i p"

            "0": "k o s o n g",
            "1": "s a t u",
            "2": "d u a",
            "3": "t i g a",
            "4": "e m p a t",
            "5": "l i m a",
            "6": "e n a m",
            "7": "t u j u h",
            "8": "d e l a p a n",
            "9": "s e m b i l a n"
        }

    def convert(self, token: str) -> str:
        # 1 Jika token utuh ada di dict, kembalikan (misal: "Rp" -> "rupiah")
        if token in self.trans_dict:
            return self.trans_dict[token]

        # 2 Jika token hanya 1 karakter (dan tidak di dict), kembalikan
        # (misal: "A" -> "A")
        if len(token) == 1:
            return token

        # 3 Jika token > 1 karakter, eja per huruf
        # (misal: ".6" -> "titik e n a m")
        return " ".join([self.convert_char(c) for c in token])

    def convert_char(self, char: str) -> str:
        # Cek dict pengeja (misal: ".")
        if char in self.trans_ordinal_dict:
            return self.trans_ordinal_dict[char]

        # Cek dict spesial (misal: "&")
        if char in self.trans_dict:
            return self.trans_dict[char]

        # Jika tidak ada, kembalikan huruf kecilnya (misal: "A" -> "a")
        return char.lower()