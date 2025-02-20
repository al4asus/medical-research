import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from scipy.interpolate import make_interp_spline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import KFold, cross_val_score

import pickle
import os
import shap

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

# print(df)

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

#hocaya sor

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


# print(list(df.columns))

df = df.drop(['kankültüründeüreyenbakteri'], axis=1)

df.fillna(value=0, inplace=True)
df[:] = np.nan_to_num(df)

# Bağımlı değişken ve bağımsız değişkenleri belirle
# df['ex_BPD'] = df['ex'] + df['BPD'] + df['rop']
X = df.drop(['ex', 'BPD','rop','bilissel6ay', 'hareket6ay','bilissel9ay','hareket9ay','bilissel12ay', 'hareket12ay', 'bilissel16ay', 'hareket16ay', 'bilissel20ay'], axis=1)
y = df['ex']

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=400)

# XGBoost modelini oluştur
#model = XGBRegressor()
# print(X_train.to_string())

#Logistic Regression
#model = LogisticRegression()

# Random Forest
#model = RandomForestClassifier()

# Support Vector Classifier
#model = SVC()

# Gradient Boosting Classifier
model = GradientBoostingClassifier()

# AdaBoost
#model = AdaBoostClassifier()

# K-Nearest Neighbor
#model = KNeighborsClassifier()

# The Decision Tree
#model = DecisionTreeClassifier()

# ExtraTrees
#model = ExtraTreesClassifier()

# Multi-Layer Perceptron Neural Network
#model = MLPClassifier()

# print(y_train)
"""
full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')

encoder = full_pipeline.fit(X_train)
X_train = encoder.transform(X_train)
"""
model.fit(X_train, y_train)



"""
print("Çalışma dizini:", os.getcwd())

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "model.pkl")  # Masaüstüne kaydet
with open(desktop_path, 'wb') as file:
    pickle.dump(model, file)
print(f"Model şu dizine kaydedildi: {desktop_path}")
"""



"""
# showing all trees with random forest method
for i, tree in enumerate(model.estimators_):
    plt.figure(figsize=(20, 10))
    plot_tree(tree, filled=True, feature_names=X.columns, class_names=["0", "1"], fontsize=10)
    plt.title(f"Decision Tree {i+1}")
    plt.show()
"""

"""
# Masaüstünde bir klasör oluştur
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
folder_path = os.path.join(desktop_path, 'gbc')

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
"""
# showing all trees with gradient boosting classifier method
"""
# Eğitim veri setinizdeki özellik isimlerini alın
feature_names = list(X_train.columns)

# Tüm ağaçları görselleştirin
for i, estimator in enumerate(model.estimators_):
    plt.figure(figsize=(19, 10))
    plot_tree(estimator[0], filled=True, feature_names=feature_names)  # feature_names parametresini ekleyin
    plt.title(f"Tree {i + 1}")
    plt.show()
"""
"""
# çalışmadı değişken adlarını yazınca hocaya sor
    # Ağacın görselini kaydet
    file_path = os.path.join(folder_path, f"ağaç_{i + 1}.png")
    plt.savefig(file_path)
    plt.close()

print("Tüm ağaç görselleri başarıyla kaydedildi.")
"""


# Modelin performansını değerlendir
y_pred = model.predict(X_test)
# print(y_test, y_pred)
y_pred_rounded = np.around(y_pred, decimals=0)

# print(y_pred_rounded)
accuracy_death = accuracy_score(y_test, y_pred_rounded)
f1_score_death = f1_score(y_test, y_pred_rounded)
precision_death = precision_score(y_test, y_pred_rounded)
sensitivity_death = recall_score(y_test, y_pred_rounded)

# Accuracy, Precision, Sensitivity (Recall), Specificity, f1
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rounded).ravel()
specificity_death = tn / (tn+fp)

print(f"Accuracy is: {accuracy_death}")
print(f"Precision is: {precision_death}")
print(f"Sensitivity is: {sensitivity_death}")
print(f"Specificity is: {specificity_death}")
print(f"f1 score is: {f1_score_death}")

print(classification_report(y_test, y_pred_rounded))

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

"""
# k fold validation
k_folds = KFold(n_splits=5)

scores = cross_val_score(model, X, y, cv=k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
"""

# Özellik önem sıralamasını al
feature_importances = model.feature_importances_

# Değişken adlarını ve önem sıralamalarını eşleştir
feature_importance_dict = dict(zip(X.columns, feature_importances))

# Önem sıralamasına göre değişkenleri sırala
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Sonuçları yazdırma kısmı
print("\nDeğişkenlerin Önem Sıralaması:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")
"""
# SHAP
shap.initjs()

explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

# Önem sıralamalarını görselleştirme kısmı
plt.figure(figsize=(15, 8))
plt.bar(range(len(feature_importances)), feature_importances, tick_label=X.columns)
plt.xlabel('Variables')
plt.ylabel('Importance Ranking')
plt.title('Variable Importance Ranking')
plt.xticks(rotation=45, ha="right")
plt.show()
"""
"""
# Area under curve
# ROC AUC skoru hesaplama kısmı
auc_score = roc_auc_score(y_test, y_pred_rounded)
print("ROC AUC Score:", auc_score)

# ROC eğrisi ve alanını hesaplayan kısım
fpr, tpr, thresholds = roc_curve(y_test, y_pred_rounded)
roc_auc = auc(fpr, tpr)

# ROC eğrisini düzgünleştirme için spline oluşturma kısmı
X_Y_Spline = make_interp_spline(fpr, tpr, k=2)  # k=2 ile ikinci dereceden spline oluşturuyoruz
fpr_smooth = np.linspace(0, 1, 1000)
tpr_smooth = X_Y_Spline(fpr_smooth)
"""

""""
# ROC eğrisi çizilen kısım
plt.figure()
plt.plot(fpr_smooth, tpr_smooth , color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 2.6])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve (ROC)')
plt.legend(loc="lower right")
plt.show()

"""
# Modeli kaydet
model_path = "trained_model.pkl"  # Kaydedilecek dosyanın adı
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

print(f"Model başarıyla '{model_path}' olarak kaydedildi.")

# Kullanılan sütunları terminale yazdır
print("\nEğitimde Kullanılan Sütunlar:")
for column in X.columns:
    print(column)