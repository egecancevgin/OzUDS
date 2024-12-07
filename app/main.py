from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import pandas as pd


with open('./app/rfx_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('./app/lencoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)


def uyumlu_ozellikler(df, model_features):
    mevcut_ozellikler = set(df.columns)
    eksik_ozellikler = set(model_features) - mevcut_ozellikler
    for feature in eksik_ozellikler:
        df[feature] = 0
    df = df[model_features]
    return df


def veri_islemesi(df, bagimli, bagimsizlar):
    """ Temel veri işlemesi yapar. 
    """
    sutunlar = bagimsizlar.copy()
    #sutunlar.append(bagimli)
    kucuk_df = df[sutunlar]
    kucuk_df[
        ["IP_s1", "IP_s2", "IP_s3", "IP_s4"]
    ] = kucuk_df["Address"].str.split(
        '.', expand=True
    )
    kucuk_df = kucuk_df.drop(columns=["IP_s1", "Address"])
    kucuk_df['Name'] = kucuk_df['Name'].str[:6]
    deneme_df = kucuk_df[[
        "IP_s2", "IP_s3", "Service", "Clus", "Dom",
        "Role", "Name", 
        "Environ"
    ]]
    deneme_df['IP_s2'] = deneme_df['IP_s2'].astype(int)
    deneme_df['IP_s3'] = deneme_df['IP_s3'].astype(int)
    #deneme_df.to_csv("env.csv", index=False)
    ddf_encoded = pd.get_dummies(
        deneme_df, columns=[
            "Service", "Environ", "Dom",
            "Role", "Name",
            "Clus"
        ]        
    )
    le = LabelEncoder()
    df_shf = ddf_encoded.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    return df_shf, le, "Contact"


def tahmin(tahmin_edilecek):
    """ Sunucu bilgilerine göre tahminleme gerçekleştirir
        :param girdi_df: DataFrame formatinda girdi
    """
    if not isinstance(tahmin_edilecek, pd.DataFrame):
        raise ValueError("Girdi verisi DataFrame formatında olmalıdır.")
    print("Bagimsiz degiskenler:", tahmin_edilecek.columns)
    bagimli_degisken = "Contact"
    bagimsiz_degiskenler = [
        "Dom", "Name", "Address", "OSDistro", "Department",
        "Virt/Phy", "Role", "Loc", "Clus",
        "Environ", "Service"         
    ]
    bagimsiz_degiskenler = tahmin_edilecek.columns
    tahmin_df, le_off, bagimli = veri_islemesi(
        tahmin_edilecek, bagimli_degisken, bagimsiz_degiskenler
    )
    model_ozellikler = model.feature_names_in_
    tahmin_df = tahmin_df.reindex(columns=model_ozellikler, fill_value=0)
    print("Model Özellikleri:", model.feature_names_in_)
    print("Tahmin Verisi Özellikleri:", tahmin_df.columns)
    tahmin_df_uyumlu = uyumlu_ozellikler(tahmin_df, model.feature_names_in_)
    tahminler = model.predict(tahmin_df_uyumlu)
    print("Tahmin sonucu:", tahminler)
    tahmin_sonucu = model.predict(tahmin_df)
    etiketler = model.classes_
    tahmin_etiketi = le.inverse_transform(tahmin_sonucu)
    print("Tahmin Etiketi:", tahmin_etiketi)    


app = FastAPI()
app.mount("/static", StaticFiles(directory="./app/static"), name="static")
templates = Jinja2Templates(directory="./app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/submit", response_class=HTMLResponse)
async def submit_form(
    request: Request,
    address: str = Form(...),
    name: str = Form(None),
    dom: str = Form(None),
    service: str = Form(None),
    role: str = Form(None),
    clus: str = Form(None),
    environ: str = Form(None)
):
    cikti_dict = {
        'Address': address,
        'Name': name,
        'Dom': dom,
        'Service': service,
        'Role': role,
        'Clus': clus,
        'Environ': environ
    }
    print(cikti_dict)
    cikti_df = pd.DataFrame([cikti_dict])
    tahmin(cikti_df)
    return templates.TemplateResponse("index.html", {"request": request})
