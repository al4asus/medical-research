import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score, roc_curve, auc
from scipy.interpolate import make_interp_spline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


excel_file_path = 'micafungindeğismemis.xlsx'  # Excel dosyasının adını belirtin
df = pd.read_excel(excel_file_path, na_values=['#BOŞ!', '#NULL!'])

# Gereksiz sütunları kaldır
df = df.drop(['bebekadi', 'dogumtarihi'], axis=1)

# 'ilac' sütununu düzeltme
df['ilac'] = df['ilac'].map({1: 'mikafungin', 2: 'fungizon', 3: 'ambizom', 4: 'flukanazol', 5: 'kaspafungin', 6: 'nistatin'})

df['gebelikhaftagunu'] = pd.read_excel('micafungindeğismemis.xlsx', usecols=['gebelikhaftagunu'])['gebelikhaftagunu']

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


def transform_difference(dataframe, column_name_1, column_name_2, column_name):
    df[column_name] = df[column_name_2] - df[column_name_1]
    # print(df[column_name])
    dataframe = dataframe.drop([column_name_1, column_name_2], axis=1)

    dataframe.loc[dataframe[column_name] < 0, f'{column_name}_decrease?'] = dataframe[column_name]
    dataframe.loc[dataframe[column_name] >= 0, f'{column_name}_decrease?'] = 0

    dataframe.loc[dataframe[column_name] < 0, f'{column_name}_increase?'] = 0
    dataframe.loc[dataframe[column_name] >= 0, f'{column_name}_increase?'] = dataframe[column_name]

    dataframe = dataframe.drop([column_name], axis=1)

    return dataframe

# Tamamlandi
# 2 deger de 40 ve alti ise hesaba katilmayacak
# aradaki fark 30'dan az ise hesaba katilmayacak
def transform_difference_ast_alt(dataframe, column_name_1, column_name_2, column_name):
    dataframe.loc[(dataframe[column_name_1] > 40) | (dataframe[column_name_2] > 40), column_name] = dataframe[column_name_2] - dataframe[column_name_1]
    dataframe.loc[(dataframe[column_name_1] <= 40) & (dataframe[column_name_2] <= 40), column_name] = 0

    dataframe.loc[(abs(dataframe[column_name]) > 30), column_name] = dataframe[column_name]
    dataframe.loc[(abs(dataframe[column_name]) <= 30), column_name] = 0

    dataframe = dataframe.drop([column_name_1, column_name_2], axis=1)

    dataframe.loc[dataframe[column_name] < 0, f'{column_name}_decrease?'] = dataframe[column_name]
    dataframe.loc[dataframe[column_name] >= 0, f'{column_name}_decrease?'] = 0

    dataframe.loc[dataframe[column_name] < 0, f'{column_name}_increase?'] = 0
    dataframe.loc[dataframe[column_name] >= 0, f'{column_name}_increase?'] = dataframe[column_name]

    dataframe = dataframe.drop([column_name], axis=1)

    return dataframe
"""
df.loc[df['proc_dif'] < 0, 'proc_dif_artmis?'] = 0
df.loc[df['proc_dif'] >= 0, 'proc_dif_artmis?'] = 1

df.loc[df['proc_dif'] < 0, 'proc_dif_azalmis?'] = 1
df.loc[df['proc_dif'] >= 0, 'proc_dif_azalmis?'] = 0
"""
# print(df)

df = transform_difference(df, "proc1", "proc2", "proc_diff")
df = transform_difference(df, "crp1", "crp2", "crp_diff")
df = transform_difference_ast_alt(df, "alt1", "alt2", "alt_diff")
df = transform_difference_ast_alt(df, "ast1", "ast2", "ast_diff")
df = transform_difference(df, "urikasit1", "urikasit2", "urikasit_diff")
df = transform_difference(df, "krea1", "krea2", "krea_diff")
df = transform_difference(df, "hb1", "hbson", "hb_diff")
df = transform_difference(df, "mpv1", "mpvson", "mpv_diff")
df = transform_difference(df, "wbc1", "wbcson", "wbc_diff")
df = transform_difference(df, "trombosit", "trombositson", "trombosit_diff")
df = transform_difference(df, "pct1", "pctson", "pct_diff")
df = transform_difference(df, "potasyum1", "potasyumson", "potasyum_diff")
df = transform_difference(df, "notrofil1", "notrofilson", "notrofil_diff")
df = transform_difference(df, "pdw1", "pdwson", "pdw_diff")
df = transform_difference(df, "ıg1", "ıgson", "ig_diff")
df = transform_difference(df, "ıl61", "ıl6son", "il6_diff")
# df = df.drop(['ast1', 'ast2'], axis=1)
# df = df.drop(['alt1', 'alt2'], axis=1)

"""
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

df = df.drop(['ıl61', 'ıl6son'], axis=1)
"""




"""
df["mrse"] = df["kankültüründeüreyenbakteri"].str.contains("mrse")
df["staph_epidermidis"] = df["kankültüründeüreyenbakteri"].str.contains("staph_epidermidis")


# Yeni bir boolean sütunu oluştur ve varsayılan olarak False atayın
df['epi_mrse'] = False

# Her bir hücredeki veri türlerini kontrol et
for index, cell_value in df['kankültüründeüreyenbakteri'].items():
    # Her bir hücreyi boşluksuz bir şekilde al
    cell_value_str = str(cell_value).strip()

    # Eğer hücrede sadece "staph epidermis" ve "mrse" bulunuyorsa, yeni sütunu True olarak işaretle
    if cell_value_str.lower() == 'staph_epidermidis,mrse' or cell_value_str.lower() == 'mrse,staph_epidermidis':
        df.at[index, 'epi_mrse'] = True





df["enterobacter"] = df["kankültüründeüreyenbakteri"].str.contains("enterobacter")
df["ecoli"] = df["kankültüründeüreyenbakteri"].str.contains("ecoli")

df['entero_eco'] = False

# Her bir hücredeki veri türlerini kontrol et
for index, a in df['kankültüründeüreyenbakteri'].items():
    # Her bir hücreyi boşluksuz bir şekilde al
    a_str = str(a).strip()

    # Eğer hücrede sadece "staph epidermis" ve "mrse" bulunuyorsa, yeni sütunu True olarak işaretle
    if a_str.lower() == 'enterobacter,ecoli' or a_str.lower() == 'ecoli,enterobacter':
        df.at[index, 'entero_eco'] = True






df["stenotrophomonas_maltophilia"] = df["kankültüründeüreyenbakteri"].str.contains("stenotrophomonas_maltophilia")




df["klebsiella_pnumonia"] = df["kankültüründeüreyenbakteri"].str.contains("klebsiella_pnumonia")
df['klebpmrse'] = False

# Her bir hücredeki veri türlerini kontrol et
for index, b in df['kankültüründeüreyenbakteri'].items():
    # Her bir hücreyi boşluksuz bir şekilde al
    b_str = str(b).strip()

    # Eğer hücrede sadece "staph epidermis" ve "mrse" bulunuyorsa, yeni sütunu True olarak işaretle
    if b_str.lower() == 'klebsiella_pnumonia,mrse' or b_str.lower() == 'mrse,klebsiella_pnumonia':
        df.at[index, 'klebpmrse'] = True






df["ralstonia_picketti"] = df["kankültüründeüreyenbakteri"].str.contains("ralstonia_picketti")

df['ralpklebp'] = False

# Her bir hücredeki veri türlerini kontrol et
for index, c in df['kankültüründeüreyenbakteri'].items():
    # Her bir hücreyi boşluksuz bir şekilde al
    c_str = str(c).strip()

    # Eğer hücrede sadece "staph epidermis" ve "mrse" bulunuyorsa, yeni sütunu True olarak işaretle
    if c_str.lower() == 'klebsiella_pnumonia,ralstonia_picketti' or c_str.lower() == 'ralstonia_picketti,klebsiella_pnumonia':
        df.at[index, 'ralpklebp'] = True





df["klebsiella"] = df["kankültüründeüreyenbakteri"].str.contains("klebsiella")
df['klebsmrse'] = False

# Her bir hücredeki veri türlerini kontrol et
for index, d in df['kankültüründeüreyenbakteri'].items():
    # Her bir hücreyi boşluksuz bir şekilde al
    d_str = str(d).strip()

    # Eğer hücrede sadece "staph epidermis" ve "mrse" bulunuyorsa, yeni sütunu True olarak işaretle
    if d_str.lower() == 'klebsiella,mrse' or d_str.lower() == 'mrse,klebsiella':
        df.at[index, 'klebsmrse'] = True






df['2mrse'] = False

# Her bir hücredeki veri türlerini kontrol et
for index, e in df['kankültüründeüreyenbakteri'].items():
    # Her bir hücreyi boşluksuz bir şekilde al
    e_str = str(e).strip()

    # Eğer hücrede sadece "staph epidermis" ve "mrse" bulunuyorsa, yeni sütunu True olarak işaretle
    if e_str.lower() == 'mrse,mrse' or e_str.lower() == 'mrse,mrse':
        df.at[index, '2mrse'] = True





df["enterokok"] = df["kankültüründeüreyenbakteri"].str.contains("enterokok")



df["streptokkous_parasanguinis"] = df["kankültüründeüreyenbakteri"].str.contains("streptokkous_parasanguinis")



df["esbl"] = df["kankültüründeüreyenbakteri"].str.contains("esbl")






df['staphe_stapha_eta'] = False

# Her bir hücredeki veri türlerini kontrol et
for index, f in df['kankültüründeüreyenbakteri'].items():
    # Her bir hücreyi boşluksuz bir şekilde al
    f_str = str(f).strip()

    # Eğer hücrede sadece "staph epidermis" ve "mrse" bulunuyorsa, yeni sütunu True olarak işaretle
    if f_str.lower() == 'staph_areus_eta,staph_epidermidis' or f_str.lower() == 'staph_epidermidis,staph_areus_eta':
        df.at[index, 'staphe_stapha_eta'] = True





df['mrse,staph_epidermidis,kateterde+'] = False

# Her bir hücredeki veri türlerini kontrol et
for index, g in df['kankültüründeüreyenbakteri'].items():
    # Her bir hücreyi boşluksuz bir şekilde al
    g_str = str(g).strip()

    # Eğer hücrede sadece "staph epidermis" ve "mrse" bulunuyorsa, yeni sütunu True olarak işaretle
    if g_str.lower() == 'mrse,staph_epidermidis,kateterde+' or g_str.lower() == 'mrse,kateterde+,staph_epidermidis,' or  g_str.lower() == 'kateterde+,mrse,staph_epidermidis' or g_str.lower() == 'kateterde+,staph_epidermidis,mrse,' or g_str.lower() == 'staph_epidermidis,kateterde+,mrse' or g_str.lower() == 'staph_epidermidis,mrse,kateterde+':
        df.at[index, 'mrse,staph_epidermidis,kateterde+'] = True





df["klebsiella_eta+kateter"] = df["kankültüründeüreyenbakteri"].str.contains("klebsiella_eta+kateter")



df["staph_hominis_periton"] = df["kankültüründeüreyenbakteri"].str.contains("staph_hominis_periton")
df['klebsiella,staph_hominis_periton'] = False

# Her bir hücredeki veri türlerini kontrol et
for index, h in df['kankültüründeüreyenbakteri'].items():
    # Her bir hücreyi boşluksuz bir şekilde al
    h_str = str(h).strip()

    # Eğer hücrede sadece "staph epidermis" ve "mrse" bulunuyorsa, yeni sütunu True olarak işaretle
    if h_str.lower() == 'klebsiella,staph_hominis_periton' or h_str.lower() == 'staph_hominis_periton,klebsiella':
        df.at[index, 'klebsiella,staph_hominis_periton'] = True






df['klebsiella,esbl+,enteroccoccus_faecium'] = False
# Her bir hücredeki veri türlerini kontrol et
for index, i in df['kankültüründeüreyenbakteri'].items():
    # Her bir hücreyi boşluksuz bir şekilde al
    i_str = str(i).strip()

    # Eğer hücrede sadece "staph epidermis" ve "mrse" bulunuyorsa, yeni sütunu True olarak işaretle
    if i_str.lower() == 'klebsiella,esbl+,enteroccoccus_faecium':
        df.at[index, 'klebsiella,esbl+,enteroccoccus_faecium'] = True






df["serratia,klebsiella,staph_eta"] = df["kankültüründeüreyenbakteri"].str.contains("serratia,klebsiella,staph_eta")
df['serratia,klebsiella,staph(eta)'] = False

# Her bir hücredeki veri türlerini kontrol et
for index, j in df['kankültüründeüreyenbakteri'].items():
    # Her bir hücreyi boşluksuz bir şekilde al
    j_str = str(j).strip()

    # Eğer hücrede sadece "staph epidermis" ve "mrse" bulunuyorsa, yeni sütunu True olarak işaretle
    if j_str.lower() == 'serratia,klebsiella,staph_eta' or j_str.lower() == 'staph_eta,serratia,klebsiella':
        df.at[index, 'serratia,klebsiella,staph_eta'] = True





df["mrse,serratia,staph_hominis"] = df["kankültüründeüreyenbakteri"].str.contains("mrse,serratia,staph_hominis")
df['mrse,serratia,staph_hominis'] = False

# Her bir hücredeki veri türlerini kontrol et
for index, k in df['kankültüründeüreyenbakteri'].items():
    # Her bir hücreyi boşluksuz bir şekilde al
    k_str = str(k).strip()

    # Eğer hücrede sadece "staph epidermis" ve "mrse" bulunuyorsa, yeni sütunu True olarak işaretle
    if k_str.lower() == 'serratia,staph_hominis,mrse' or k_str.lower() == 'mrse,serratia,staph_hominis':
        df.at[index, 'mrse,serratia,staph_hominis'] = True






df["klebsiella_idrarda,streptococcus_mitis_oralis_idrarda,candida"] = df["kankültüründeüreyenbakteri"].str.contains("klebsiella_idrarda,streptococcus_mitis_oralis_idrarda,candida")
df['klebsiella_idrarda,streptococcus_mitis_oralis_idrarda,candida'] = False

# Her bir hücredeki veri türlerini kontrol et
for index, l in df['kankültüründeüreyenbakteri'].items():
    # Her bir hücreyi boşluksuz bir şekilde al
    l_str = str(l).strip()

    # Eğer hücrede sadece "staph epidermis" ve "mrse" bulunuyorsa, yeni sütunu True olarak işaretle
    if l_str.lower() == 'klebsiella_idrarda,streptococcus_mitis_oralis_idrarda,candida' or l_str.lower() == 'candida,klebsiella_idrarda,streptococcus_mitis_oralis_idrarda':
        df.at[index, 'klebsiella_idrarda,streptococcus_mitis_oralis_idrarda,candida'] = True






df["serratia_marcasens"] = df["kankültüründeüreyenbakteri"].str.contains("serratia_marcasens")




df["klebsiella,staph_capitis,stenotrophomonas_maltophilia,rothia_mucilaginosa_pertion"] = df["kankültüründeüreyenbakteri"].str.contains("klebsiella,staph_capitis,stenotrophomonas_maltophilia,rothia_mucilaginosa_pertion")
df['klebsiella,staph_capitis,stenotrophomonas_maltophilia,rothia_mucilaginosa_pertion'] = False

# Her bir hücredeki veri türlerini kontrol et
for index, m in df['kankültüründeüreyenbakteri'].items():
    # Her bir hücreyi boşluksuz bir şekilde al
    m_str = str(m).strip()

    # Eğer hücrede sadece "staph epidermis" ve "mrse" bulunuyorsa, yeni sütunu True olarak işaretle
    if m_str.lower() == 'klebsiella,staph_capitis,stenotrophomonas_maltophilia,rothia_mucilaginosa_pertion' or m_str.lower() == 'klebsiella,staph_capitis,stenotrophomonas_maltophilia,rothia_mucilaginosa_pertion':
        df.at[index, 'klebsiella,staph_capitis,stenotrophomonas_maltophilia,rothia_mucilaginosa_pertion'] = True


"""


print(list(df.columns))

df = df.drop(['kankültüründeüreyenbakteri'], axis=1)

# Bağımlı değişken ve bağımsız değişkenleri belirle
# df['ex_BPD'] = df['ex'] + df['BPD'] + df['rop']
X = df.drop(['ex', 'BPD','rop','bilissel6ay', 'hareket6ay','bilissel9ay','hareket9ay','bilissel12ay', 'hareket12ay', 'bilissel16ay', 'hareket16ay', 'bilissel20ay'], axis=1)
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
y_pred_rounded = np.around(y_pred, decimals=0)

print(y_pred_rounded)
accuracy_death = accuracy_score(y_test, y_pred_rounded)
f1_score_death = f1_score(y_test, y_pred_rounded)

print(f"Accuracy is: {accuracy_death}")
print(f"f1 score is: {f1_score_death}")

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

plt.clf()

# Area under curve
# ROC AUC skoru hesaplama
auc_score = roc_auc_score(y_test, y_pred_rounded)
print("ROC AUC Score:", auc_score)

print(df["ast_diff_increase?"])
print(df["ast_diff_decrease?"])

# ROC eğrisi ve alanını hesapla
fpr, tpr, thresholds = roc_curve(y_test, y_pred_rounded)
roc_auc = auc(fpr, tpr)


# ROC eğrisini düzgünleştirme için spline oluşturma
X_Y_Spline = make_interp_spline(fpr, tpr, k=2)  # k=2 ile ikinci dereceden spline oluşturuyoruz
fpr_smooth = np.linspace(0, 1, 1000)
tpr_smooth = X_Y_Spline(fpr_smooth)

"""
fpr_inside_bounds = []
tpr_inside_bounds = []
for f, t in zip(fpr_smooth, tpr_smooth):
    if 0 <= f <= 1 and 0 <= t <= 1:  # x ve y koordinatlarını kontrol et
        fpr_inside_bounds.append(f)
        tpr_inside_bounds.append(t)
"""
# ROC eğrisini çiz
plt.figure()
plt.plot(fpr_smooth, tpr_smooth , color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.5])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Smooth Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()