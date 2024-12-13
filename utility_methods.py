from collections import Counter
import pandas as pd
import numpy as np


# Global degiskenler
karaliste_df = pd.read_csv("./veri-setleri/karaliste.csv", dtype=str)
karaliste = karaliste_df["Sicille"].tolist()
sicil_ekip = pd.read_csv("./veri-setleri/sicil-ekip.csv")
sicil_ekip_sozluk = dict(zip(sicil_ekip["Sicille"], sicil_ekip["Ekiple"]))
uygulama_surec = [
    "java", "nagios", "docker", "elasticsearch", "k8s"
]


def excel_envanter_oku(excelfile):
    """ Girdide verilen .xlsx envanter dosyasını okur
        :param excelfile: Excel dosyasının yolu
    """
    df = pd.read_excel(excelfile)
    df = df[[
        "Dom"
    ]]
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    shuffled_df.to_csv("6kk_env.csv", index=False)


def unique_ekipleri_al(csvfile):
    """ Girdideki unique Contact'leri ekipler.csv'ye atar
    :param csvfile: Girdi .csv dosyası
    returns: Çıktı .csv dosyası olan ekipler.csv
    """
    df = pd.read_csv("6k_env.csv")
    unique_contacts = sorted(df["Contact"].unique())
    with open("ekipler.csv", "w") as dosya:
        for contact in unique_contacts:
            dosya.write(f"{contact}\n")
    return "ekipler.csv"


def appcontact_normalizasyonu(csvfile):
    """ Contact sütunlarını sadeleştirir.
    :param csvfile: Girdi .csv envanter dosyası
    """
    df = pd.read_csv(csvfile)

    df["Contact"] = df["Contact"].astype(str).str.split(
        "@fake", n=1
    ).str[0]
    df["Contact"] = df["Contact"].astype(str).str.split(
        "@fake2fake", n=1
    ).str[0]
    df['Contact'] = df[
        'Contact'
    ].str.split('<').str[0].str.strip().str.strip('."')
    df['Contact'] = df['Contact'].str.lower()
    df.to_csv("normalize_env.csv", index=False)


def ekiplerin_normalizasyonu(csvfile):
    """ Ekip isimlerini normalize eder
        :param csvfile: .csv uzantılı girdi dosyası
    """
    df['Ekip İsmi'] = df['Ekip İsmi'].str.lower()
    df.to_csv("./manuel_sicil/Nsuheyp.csv", index=False)


def appcontact_kucukharf(csvfile):
    """ Envanterdeki ekip isimlerini küçük harf yapar
        :param csvfile: Envanter csv dosyası
    """
    df = pd.read_csv(csvfile)
    df['Contact'] = df['ApplicationContact'].str.lower()
    df.to_csv("kharf_mail_env.csv", index=False)


def sicilno_ayrisimi(log):
        """ Last komutundan sadece sicil numaralarini alan filtre
            :param log: Bir satir Lastlog verisi
        """
        lines = log.split('\n')
        sicil_numbers = [line.split(':')[0].strip() for line in lines]
        return ','.join(sicil_numbers)


def karaliste_filtresi(sicil_string):
    """ Harici sicillerde olanları LastSicilleri sütunundan çıkarır
        :param sicil_string: Sicillerin bulunduğu filtrelenmiş satır
    """
    sicil_list = [sicil.strip() for sicil in sicil_string.split(',')]
    filt_list = []
    for sicil in sicil_list:
        if sicil in karaliste:
            #filt_list.append("Harici")
            pass
        else:
            filt_list.append(sicil)
    return ','.join(filt_list)


def sicillerden_ekiplere(sicil_string):
    """ Normalize last ciktisina uygulanır, sicilden ekibe eşleştirir
        :param sicil_string: Sicilleri ekiplere eşitler 
    """
    sicil_list = [sicil.strip() for sicil in sicil_string.split(",")]
    ekip_list = [
        sicil_ekip_sozluk[sicil] if sicil in sicil_ekip_sozluk else "Bilinmiyor" for sicil in sicil_list
    ]
    k = ",".join(ekip_list)
    return k


def en_cok_giren_ekip(ekip_string):
    """ Ciktida en fazla karsilasilan ekibi döner
        :param ekip_string: Ekiplerin bulunduğu metin
    """
    ekip_list = [ekip.strip() for ekip in ekip_string.split(",")]
    ekip_sayac = Counter(ekip_list)
    en_cok_giren_ekip = ekip_sayac.most_common(1)[0][0]
    return en_cok_giren_ekip
    

def last_normalizasyonu(csvdosyasi):
    """ Last komutu ciktisini normalize eder
        :param csvdosyasi: Lastlog ve ps ciktili envanter dosyasi
        :param siciller: Sicil, Ekip sütunları olan .csv dosyasi
    """
    df = pd.read_csv(csvdosyasi)
    df["LastSicilleri"] = df['Lastlog'].apply(sicilno_ayrisimi)
    df['LastSicilleri'] = df['LastSicilleri'].apply(karaliste_filtresi)
    df["Last-Ekip"] = df["LastSicilleri"].apply(sicillerden_ekiplere)
    df["Sık-Giren-Ekip"] = df["Last-Ekip"].apply(en_cok_giren_ekip)
    df[[
        "IP", "Last-Ekip", "LastSicilleri", "Sık-Giren-Ekip"
    ]].to_csv("DENEME.csv", index=False)


def process_normalizasyonu(csvdosyasi, uygulama_surec):
    """ Çalışan root harici süreçleri normalize eder
        :param csvdosyasi: Envanter dosyasi
    """
    df = pd.read_csv(csvdosyasi)
    harici_secenekler = ['-u', '-t', '-l', '--', '-f', '-w', '--syst', '-']
    df["Parsed Process"] = df["Process"].apply(
        lambda x: [item.strip('-') for item in str(x).split() if item not in harici_secenekler] if pd.notnull(x) else []
    )
    df["Uyg Sureci"] = df["Parsed Process"].apply(
        lambda process_list: ','.join(set(
            item for item in process_list if any(keyword in item for keyword in uygulama_surec)
        ))
    )
    print(df["Uyg Sureci"][0])
    df[["IP", "Hostname", "Uyg Sureci"]].to_csv("./DENEME.csv", index=False)


def appc_mail_normalizasyonu(csvdosyasi):
    """ 
        Contact sütunundaki mailleri hedef yapar
        :param csvdosyasi: Girdi .csv dosyasi
    """
    df = pd.read_csv(csvdosyasi)
    # AppContact sütunundaki degeri mail yapar
    df['Contact'] = np.where(
        df['Contact'].str.split('<').str[0].str.contains('@'),
        df['Contact'].str.split('<').str[0].str.strip(),
        df['Contact'].str.split('<').str[1]
    )
    remove_list = ['yyekibi', ';', '>']
    for item in remove_list:
        df['Contact'] = df['Contact'].str.replace(item, '', regex=False)
    df.to_csv("./veri-setleri/mailli_env.csv", index=False)


def environment_normalizasyonu(csvdosyasi):
    """ Environment sutununda gerekli iyileştirmeler yapilir
        Ör: PRODUCTION olanları PROD yapar ki tutarlı olsun
        :param csvdosyasi: CSV formatindaki envanter dosyasi
    """
    df = pd.read_csv(csvdosyasi)
    df["Environ"] = df["Environ"].replace("PRODUCTION", "PROD")
    df["Environ"] = df["Environ"].replace("PREPRODUCTION", "PREPROD")
    df.to_csv("./veri-setleri/prodmail_env.csv", index=False)


def sahipsizler(csvdosyasi):
    """ Sahipsiz sunucuları tespit eder 
        :param csvdosyasi: Envanter dosyasi
    """
    df = pd.read_csv(csvdosyasi)
    pass


def main():
    """ Surucu fonksiyon """
    #excel_envanter_oku("unix_env.xlsx")
    #unique_ekipleri_al("6k_env.csv")
    #appcontact_normalizasyonu("6k_env.csv")
    #ekiplerin_normalizasyonu("./manuel_sicil/suheyp.csv")
    #appcontact_kucukharf("normalize_env.csv")
    #last_normalizasyonu("./veri-setleri/komut_ciktilari.csv", 0)
    #appc_mail_normalizasyonu("6k_env.csv")
    #appc_mail_normalizasyonu("kharf_mail_env.csv")
    #environment_normalizasyonu("./veri-setleri/mailli_env.csv")
    process_normalizasyonu("./veri-setleri/komut_ciktilari.csv", uygulama_surec)


if __name__ == "__main__":
    main()
