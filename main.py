import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Excel dosyasını oku, #BOŞ! ve #NULL! değerleri NaN olarak işlenecek
excel_file_path = 'micafungin.xlsx'  # Excel dosyasının adını belirtin
df = pd.read_excel(excel_file_path, na_values=['#BOŞ!', '#NULL!'])

# Gereksiz sütunları kaldır
df = df.drop(['bebekadi', 'dogumtarihi'], axis=1)

# 'ilac' sütununu düzeltme
df['ilac'] = df['ilac'].map({1: 'mikafungin', 2: 'fungizon', 3: 'ambizom', 4: 'flukanazol', 5: 'kaspafungin', 6: 'nistatin'})

df['gebelikhaftagunu'] = pd.read_excel('micafungin.xlsx', usecols=['gebelikhaftagunu'])['gebelikhaftagunu']

def categorize_gebelik_haftasi(gebelik_haftasi):
    gruplar = [[], [], [], [], []]
    if gebelik_haftasi < 28:
        gruplar[0] += [gebelik_haftasi]
    elif 28 <= gebelik_haftasi < 32:
        gruplar[1] += [gebelik_haftasi]
    elif 32 <= gebelik_haftasi < 34:
        gruplar[2] += [gebelik_haftasi]
    elif 34 <= gebelik_haftasi < 37:
        gruplar[3] += [gebelik_haftasi]
    else:
        gruplar[4] += [gebelik_haftasi]

df['gebelik_haftasi_gruplu'] = df['gebelikhaftasi'].apply(categorize_gebelik_haftasi)

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['antifungalkullanimi'])

# antifungalkullanimi sütununu kategorik olarak işaretle
df['antifungalkullanimi'] = df['antifungalkullanimi'].astype(int)

# print(df["antifungalkullanimi"].astype("category"))
# df['antifungalkullanimi'] = df['antifungalkullanimi'].astype('category')

print(df)

# Kategorik değişkenleri one-hot encoding ile dönüştür
df = pd.get_dummies(df, columns=['ilac', 'gebelik_haftasi_gruplu', 'tani_gruplu', 'dogum_agirligi_gruplu', 'cinsiyeti',
                                 'gebelik_tipi_gruplu', 'dogumsekli', 'nefrotoksikilacne', 'antifungalkullanimi', 'tanisi'], drop_first=True)

"""
df['antifungalkullanimi'] = df['antifungalkullanimi'].map({
    0.0: 'Yok',
    1.0: 'Kantiseptik Kullanmış',
    2.0: 'Amfetorisin Kullanmış',
    3.0: 'Flukazanol Kullanmış',
    4.0: 'Mikafungin Kullanmış',
    5.0: 'Kaspafungin Kullanmış'
})
"""
# NaN değerlere sahip satırları atla
df = df.fillna(method="ffill")

df['proc_dif'] = df['proc2'] - df['proc1']
print(df['proc_dif'])
df = df.drop(['proc1', 'proc2'], axis=1)

df['crp_dif'] = df['crp2'] - df['crp1']
print(df['crp_dif'])
df = df.drop(['crp1', 'crp2'], axis=1)

df['alt_dif'] = df['alt2'] - df['alt1']
print(df['alt_dif'])
df = df.drop(['alt1', 'alt2'], axis=1)

df['ast_dif'] = df['ast2'] - df['ast1']
print(df['ast_dif'])
df = df.drop(['ast1', 'ast2'], axis=1)

df['urikasit_dif'] = df['urikasit2'] - df['urikasit1']
print(df['urikasit_dif'])
df = df.drop(['urikasit1', 'urikasit2'], axis=1)

df['krea_dif'] = df['krea2'] - df['krea1']
print(df['krea_dif'])
df = df.drop(['krea1', 'krea2'], axis=1)

df['hb_dif'] = df['hbson'] - df['hb1']
print(df['hb_dif'])
df = df.drop(['hb1', 'hbson'], axis=1)

df['mpv_dif'] = df['mpvson'] - df['mpv1']
print(df['mpv_dif'])
df = df.drop(['mpv1', 'mpvson'], axis=1)

df['wbc_dif'] = df['wbcson'] - df['wbc1']
print(df['wbc_dif'])
df = df.drop(['wbc1', 'wbcson'], axis=1)

df['trombosit_dif'] = df['trombositson'] - df['trombosit']
print(df['trombosit_dif'])
df = df.drop(['trombosit', 'trombositson'], axis=1)

df['pct_dif'] = df['pctson'] - df['pct1']
print(df['pct_dif'])
df = df.drop(['pct1', 'pctson'], axis=1)

df['potasyum_dif'] = df['potasyumson'] - df['potasyum1']
print(df['potasyum_dif'])
df = df.drop(['potasyum1', 'potasyumson'], axis=1)

df['notrofil_dif'] = df['notrofilson'] - df['notrofil1']
print(df['notrofil_dif'])
df = df.drop(['notrofil1', 'notrofilson'], axis=1)

df['pdw_dif'] = df['pdwson'] - df['pdw1']
print(df['pdw_dif'])
df = df.drop(['pdw1', 'pdwson'], axis=1)

df['ıg_dif'] = df['ıgson'] - df['ıg1']
print(df['ıg_dif'])
df = df.drop(['ıg1', 'ıgson'], axis=1)

df['ıl_dif'] = df['ıl6son'] - df['ıl61']
print(df['ıl_dif'])
df = df.drop(['ıl61', 'ıl6son'], axis=1)












print(list(df.columns))

# Bağımlı değişken ve bağımsız değişkenleri belirle
# df['ex_BPD'] = df['ex'] + df['BPD'] + df['rop']
X = df.drop(['ex', 'BPD', 'rop'], axis=1)
y = df['ex']

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=400)

# XGBoost modelini oluştur
model = XGBRegressor()
print(X_train.to_string())
# print(y_train)
"""
full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')

encoder = full_pipeline.fit(X_train)
X_train = encoder.transform(X_train)
"""
model.fit(X_train, y_train)

# Modelin performansını değerlendir
y_pred = model.predict(X_test)
print(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Özellik önem sıralamasını al
feature_importances = model.feature_importances_

# Değişken adlarını ve önem sıralamalarını eşleştir
feature_importance_dict = dict(zip(X.columns, feature_importances))

# Önem sıralamasına göre değişkenleri sırala
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Sonuçları yazdır
print("\nDeğişkenlerin Önem Sıralaması:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")

# Önem sıralamalarını görselleştir
plt.figure(figsize=(15, 8))
plt.bar(range(len(feature_importances)), feature_importances, tick_label=X.columns)
plt.xlabel('Değişkenler')
plt.ylabel('Önem Sıralaması')
plt.title('Değişken Önem Sıralaması')
plt.xticks(rotation=45, ha="right")
plt.show()
