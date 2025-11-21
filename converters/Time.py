from singleton_decorator import singleton
import re
from .Cardinal import Cardinal

@singleton
class Time:
    """
    Class ini mengkonversi format waktu (jam) dan durasi
    ke dalam bentuk terucap Bahasa Indonesia.
    
    Contoh Jam:
    "14:30" -> "empat belas tiga puluh"
    "2:30 pm" -> "dua tiga puluh p m"
    "2:00" -> "dua tepat"
    "14:00" -> "empat belas tepat"
    "14:05" -> "empat belas kosong lima"
    
    Contoh Durasi:
    "01:30:45" -> "satu jam tiga puluh menit dan empat puluh lima detik"
    """
    def __init__(self):
        super().__init__()

        # Regex
        self.filter_regex = re.compile(r"[. ]")
        self.time_regex = re.compile(r"^(?P<hour>\d{1,2}) *((?::|.) *(?P<minute>\d{1,2}))? *(?P<suffix>[a-zA-Z\. ]*)$", flags=re.I)
        self.full_time_regex = re.compile(r"^(?:(?P<hour>\d{1,2}) *:)? *(?P<minute>\d{1,2})(?: *: *(?P<seconds>\d{1,2})(?: *. *(?P<milliseconds>\d{1,2}))?)? *(?P<suffix>[a-zA-Z\. ]*)$", flags=re.I)
        self.ampm_time_regex = re.compile(r"^(?P<suffix>[a-zA-Z\. ]*)(?P<hour>\d{1,2})", flags=re.I)

        self.cardinal = Cardinal()

    def convert(self, token: str) -> str:

        # 1 Strip spasi
        token = token.strip()
        result_list = []

        # 2 Coba match "hh.mm (pm)" (Format Jam)
        match = self.time_regex.match(token)
        if match:
            hour, minute, suffix = match.group("hour"), match.group("minute"), match.group("suffix")
            ampm = self.filter_regex.sub("", suffix).lower().startswith(("am", "pm"))

            # 2.1 Konversi jam (gunakan modulo jika ada am/pm)
            if ampm:
                result_list.append(self.cardinal.convert(self.modulo_hour(hour)))
            else:
                result_list.append(self.cardinal.convert(hour))

            # 2.2 Tambahkan menit jika ada dan bukan "00"
            if minute and minute != "00":
                if minute[0] == "0":
                    result_list.append("kosong") # "o" -> "kosong"
                result_list.append(self.cardinal.convert(minute))

            elif not ampm:
                # 2.3 Jika tidak ada menit (00) dan tidak ada am/pm
                # "hundred" atau "o'clock" diganti "tepat"
                result_list.append("tepat")
                
            # 2.4 Tambahkan sufiks (am/pm dieja)
            if suffix:
                result_list += [c for c in suffix.lower() if c not in (" ", ".")]
            
            return " ".join(result_list)

        # 3 Coba match "(hh:)mm:ss(.ms) (pm)" (Format Durasi)
        match = self.full_time_regex.match(token)
        if match:
            hour, minute, seconds, milliseconds, suffix = (
                match.group("hour"), match.group("minute"), match.group("seconds"), 
                match.group("milliseconds"), match.group("suffix")
            )
            
            # 3.1 Jam
            if hour:
                result_list.append(self.cardinal.convert(hour))
                result_list.append("jam") # "hour(s)" -> "jam"
            # 3.2 Menit
            if minute:
                result_list.append(self.cardinal.convert(minute))
                result_list.append("menit") # "minute(s)" -> "menit"
            # 3.3 Detik
            if seconds:
                if not milliseconds:
                    result_list.append("dan") # "and" -> "dan"
                result_list.append(self.cardinal.convert(seconds))
                result_list.append("detik") # "second(s)" -> "detik"
            # 3.4 Milidetik
            if milliseconds:
                result_list.append("dan") # "and" -> "dan"
                result_list.append(self.cardinal.convert(milliseconds))
                result_list.append("milidetik") # "millisecond(s)" -> "milidetik"
            
            # 3.5 Sufiks
            if suffix:
                result_list += [c for c in suffix.lower() if c not in (" ", ".")]
            
            return " ".join(result_list)
        
        # 4 Coba match "xxH" (misal "PM3")
        match = self.ampm_time_regex.match(token)
        if match:
            hour, suffix = match.group("hour"), match.group("suffix")
            ampm = self.filter_regex.sub("", suffix).lower().startswith(("am", "pm"))

            if ampm:
                result_list.append(self.cardinal.convert(self.modulo_hour(hour)))
            else:
                result_list.append(self.cardinal.convert(hour))
            result_list += [c for c in suffix.lower() if c not in (" ", ".")]
            return " ".join(result_list)

        return token

    def modulo_hour(self, hour: str) -> str:
        if hour == "12":
            return hour
        return str(int(hour) % 12)