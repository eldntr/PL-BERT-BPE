from singleton_decorator import singleton
import re
from .Cardinal import Cardinal
# from .Ordinal import Ordinal 

@singleton
class Date:
    """
    Class ini mengkonversi berbagai format tanggal ke dalam bentuk
    terucap dalam Bahasa Indonesia.
    
    Contoh:
    "5 Mei" -> "lima mei"
    "Senin, 17-08-1945" -> "senin tujuh belas agustus seribu sembilan ratus empat puluh lima"
    "2000s" -> "dua ribuan"
    "1990s" -> "seribu sembilan ratus sembilan puluhan"
    "13 SM" -> "tiga belas sebelum masehi"
    """
    def __init__(self):
        super().__init__()
        # Regex untuk filter koma dan apostrof
        self.filter_regex = re.compile(r"[,']")
        
        # Regex untuk hari
        self.day_regex = re.compile(
            r"^(?P<prefix>senin|selasa|rabu|kamis|jumat|sabtu|minggu|sen|sel|rab|kam|jum|sab|min|mgg)\.?", 
            flags=re.I
        )

        # Daftar bulan untuk regex
        self.month_list_regex = (
            r"(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember|"
            r"jan|feb|mar|apr|jun|jul|agu|sep|sept|okt|nov|des)"
        )

        # Regex untuk format tanggal DD-MM-YYYY
        self.dash_date_dmy_regex = re.compile(r"^(?P<day>\d{1,2}) *(?:-|\.|/) *(?P<month>\d{1,2}) *(?:-|\.|/) *(?P<year>\d{2,5})$", flags=re.I)
        # Regex untuk yyyy-mm-dd (ISO)
        self.dash_date_ymd_regex = re.compile(r"^(?P<year>\d{2,5}) *(?:-|\.|/) *(?P<month>\d{1,2}) *(?:-|\.|/) *(?P<day>\d{1,2})$", flags=re.I)
        # Regex untuk mm-dd-yyyy (US)
        self.dash_date_mdy_regex = re.compile(r"^(?P<month>\d{1,2}) *(?:-|\.|/) *(?P<day>\d{1,2}) *(?:-|\.|/) *(?P<year>\d{2,5})$", flags=re.I)

        # Regex untuk tanggal dengan bulan teks
        self.text_ymd_regex = re.compile(r"^(?P<year>\d{2,5}) *(?:-|\.|/) *" + self.month_list_regex + r" *(?:-|\.|/) *(?P<day>\d{1,2})$", flags=re.I)
        self.text_dmy_regex = re.compile(r"^(?P<day>\d{1,2}) *(?:-|\.|/) *" + self.month_list_regex + r" *(?:-|\.|/) *(?P<year>\d{2,5})$", flags=re.I)
        self.text_mdy_regex = re.compile(r"^(?P<month>" + self.month_list_regex + r") *(?:-|\.|/) *(?P<day>\d{1,2}) *(?:-|\.|/) *(?P<year>\d{2,5})$", flags=re.I)

        # Regex untuk format "DD Bulan YYYY", "Bulan YYYY", "YYYY", "YYYYs"
        self.dmy_regex = re.compile(r"^(?:(?:(?P<day>\d{1,2}) +)?" + self.month_list_regex + r"\.? +)?(?P<year>\d{1,5})(?P<suffix>s?)\/?(?: *(?P<bcsuffix>[A-Z\.]+)?)$", flags=re.I)
        # Regex untuk "Bulan DD, YYYY"
        self.mdy_regex = re.compile(r"^(?P<month>" + self.month_list_regex + r")?\.? *(?P<day>\d{1,2})? +(?P<year>\d{1,5})(?P<suffix>s?)\/?(?: *(?P<bcsuffix>[A-Z\.]+)?)$", flags=re.I)

        # Regex untuk "DD Bulan"
        self.dm_regex = re.compile(r"^(?P<day>\d{1,2}) +(?P<month>" + self.month_list_regex + r")\.?(?: *(?P<bcsuffix>[A-Z\.]+)?)$", flags=re.I)
        # Regex untuk "Bulan DD"
        self.md_regex = re.compile(r"^(?P<month>" + self.month_list_regex + r")\.? +(?P<day>\d{1,2})(?: *(?P<bcsuffix>[A-Z\.]+)?)$", flags=re.I)

        # Regex untuk hapus "th", "nd", "st" (eg. input "5th Mei")
        self.th_regex = re.compile(r"(?:(?<=\d)|(?<=\d ))(?:th|nd|rd|st)", flags=re.I)

        # Dict bulan
        self.trans_month_dict = {
            "jan": "januari",
            "feb": "februari",
            "mar": "maret",
            "apr": "april",
            "mei": "mei",
            "jun": "juni",
            "jul": "juli",
            "agu": "agustus",
            "sep": "september",
            "sept": "september",
            "okt": "oktober",
            "nov": "november",
            "des": "desember",
            
            "01": "januari", "1": "januari",
            "02": "februari", "2": "februari",
            "03": "maret", "3": "maret",
            "04": "april", "4": "april",
            "05": "mei", "5": "mei",
            "06": "juni", "6": "juni",
            "07": "juli", "7": "juli",
            "08": "agustus", "8": "agustus",
            "09": "september", "9": "september",
            "10": "oktober",
            "11": "november",
            "12": "desember",
        }

        # Dict hari
        self.trans_day_dict = {
            "sen": "senin",
            "sel": "selasa",
            "rab": "rabu",
            "kam": "kamis",
            "jum": "jumat",
            "sab": "sabtu",
            "min": "minggu",
            "mgg": "minggu",
        }

        # Konverter Cardinal
        self.cardinal = Cardinal()
    
    def convert(self, token: str) -> str:
        # Variabel untuk menyimpan bagian-bagian tanggal
        prefix = None
        day = None
        month = None
        year = None
        suffix = None

        # 1.1 Hapus koma, dll.
        token = self.filter_regex.sub("", token).strip()

        # 1.2 Hapus "th", "nd", "st"
        match = self.th_regex.search(token)
        if match:
            token = token[:match.span()[0]] + token[match.span()[1]:]

        # 1.3 Cek prefix hari (misal "Senin 14 Mei 2009")
        match = self.day_regex.match(token)
        if match:
            prefix = self.get_prefix(match.group("prefix"))
            token = token[match.span()[1]:].strip()

        # 1.4 Hapus prefix "tanggal " (versi Indonesia dari "the ")
        if token.lower().startswith("tanggal "):
            token = token[8:].strip()

        # Fungsi helper untuk membangun output (format Indonesia: HARI TANGGAL BULAN TAHUN)
        def construct_output():
            result_list = []
            result_list.append(prefix)
            result_list.append(day)
            result_list.append(month)
            result_list.append(year)
            result_list.append(suffix)
            # Gabungkan semua bagian yang tidak kosong
            return " ".join([result for result in result_list if result])

        # 2 Match "DD Bulan" atau "Bulan DD"
        match = self.dm_regex.match(token)
        if not match:
            match = self.md_regex.match(token)
        
        if match:
            # Gunakan CARDINAL ("lima") bukan ORDINAL ("kelima") karena format tanggal di Indonesia
            day = self.cardinal.convert(match.group("day")) 
            month = self.get_month(match.group("month"))
            try:
                suffix = self.get_suffix(match.group("bcsuffix"))
            except (IndexError, AttributeError):
                pass
            return construct_output()

        # 3 Match format DD-MM-YYYY, YYYY-MM-DD, MM-DD-YYYY, atau versi teks
        # Prioritaskan D-M-Y (Indonesia) -> Y-M-D (ISO) -> M-D-Y (US)
        match = (
            self.dash_date_dmy_regex.match(token) or
            self.text_dmy_regex.match(token) or
            self.dash_date_ymd_regex.match(token) or
            self.text_ymd_regex.match(token) or
            self.dash_date_mdy_regex.match(token) or
            self.text_mdy_regex.match(token)
        )
        if match:
            group_dict = match.groupdict()
            day_num = group_dict.get("day")
            month_num = group_dict.get("month")
            year_num = group_dict.get("year")
            
            # Coba tukar jika formatnya jelas salah (misal 31-01-2000 tapi ter-match sbg MM-DD)
            try:
                if month_num is not None and int(month_num) > 12:
                    month_num, day_num = day_num, month_num
            except ValueError:
                # Bulan adalah teks (misal "januari"), tidak perlu ditukar
                pass
            except TypeError:
                # Salah satu nilai None, abaikan
                pass

            if month_num:
                month = self.get_month(month_num)
            year = self.convert_year(year_num)
            if day_num:
                # Gunakan CARDINAL
                day = self.cardinal.convert(day_num) 
            return construct_output()

        # 4 Match "DD Bulan YYYY", "Bulan YYYY", "YYYY", "YYYYs"
        match = self.dmy_regex.match(token)
        if not match:
            match = self.mdy_regex.match(token)
        
        if match:
            group_dict = match.groupdict()
            day_value = group_dict.get("day")
            month_token = group_dict.get("month")
            if day_value:
                # Gunakan CARDINAL
                day = self.cardinal.convert(day_value) 
            if month_token:
                month = self.get_month(month_token)
            
            # Cek apakah ini dekade (misal "2000s")
            if group_dict.get("suffix"):
                year = self.convert_year(group_dict.get("year"), cardinal=False) # cardinal=False -> tambahkan "-an"
            else:
                year = self.convert_year(group_dict.get("year"))
            
            try:
                suffix = self.get_suffix(group_dict.get("bcsuffix"))
            except (IndexError, AttributeError):
                pass
            return construct_output()

        return token

    def get_prefix(self, prefix: str) -> str:
        # Helper untuk normalisasi nama hari
        if prefix is None:
            return prefix
        if prefix.lower() in self.trans_day_dict:
            return self.trans_day_dict[prefix.lower()]
        return prefix.lower()

    def get_suffix(self, suffix_str: str) -> str:
        # Helper untuk melokalkan BC/AD
        if suffix_str is None:
            return None
        
        s = suffix_str.lower().replace(".", "")
        if s == "ad" or s == "m":
            return "masehi"
        if s == "bc" or s == "sm":
            return "sebelum masehi"
        
        # Fallback jika tidak dikenal (misal "CE")
        return " ".join([c for c in s if c != " "])

    def convert_year(self, token: str, cardinal:bool = True) -> str:
        # Tahun di Indonesia dibaca sebagai angka kardinal penuh.
        
        # "00" -> "kosong kosong" (misal '00)
        if token == "00":
            return "kosong kosong"

        # Konversi seluruh tahun sebagai satu angka kardinal
        # misal "1945" -> "seribu sembilan ratus empat puluh lima"
        # misal "2001" -> "dua ribu satu"
        result = self.cardinal.convert(token)

        # Jika bukan cardinal, berarti ini dekade (misal "1990s" atau "2000s")
        # Kita perlu menambahkan sufiks "-an"
        if not cardinal:
            # "sepuluh" -> "sepuluhan"
            # "dua puluh" -> "dua puluhan"
            if result.endswith("h"): 
                result = result[:-1] + "han" 
            # "dua ribu" -> "dua ribuan"
            # "enam" -> "enaman" (misal '60s -> "enam puluhan")
            else:
                result += "an"
        
        return result

    def get_month(self, token: str) -> str:
        # Helper untuk normalisasi nama bulan
        if not token:
            return token
        if token.lower() in self.trans_month_dict:
            return self.trans_month_dict[token.lower()]
        return token.lower()