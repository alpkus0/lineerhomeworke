import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# Veri kümesini yükleyelim
database = load_diabetes()

# Kümeyi DF'e çevirelim
df = pd.DataFrame(data=database.data, columns=database.feature_names)

# Hedef değişkenimizi belirleyelim
df['target'] = database.target

print("\nLineer Regresyon Veri Seti:")
print(df.head())

# Model Eğitme Basit Lineer Regresyon
x = df[['bmi']]  # 'bmi' sütununu bir DataFrame formatında seçiyoruz, çünkü model bir DataFrame bekliyor
y = df['target']

x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"\nEğitim Seti Boyutu: {x_train.shape}")
print(f"Test Seti Boyutu: {x_test.shape}")

print("\nBasit Lineer Regresyon Modeli Eğitiliyor...")

model = LinearRegression()
model.fit(x_train, y_train)

print("Basit Lineer Regresyon Modeli Eğitildi!\n")
print("Katsayısal:")
print(model.coef_)
print(f"Intercept (b0) : {model.intercept_}")

# r2 Skoru
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(f"\nBasit Lineer Regresyon r2 skoru: {r2:.4f}\n")

print("------------------------------------------------------------------------------")
# Çoklu Regresyon
print("\nÇoklu Lineer Regresyon\n")

# Model Eğitme
new_x = df[['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']]
new_y = df['target']

new_x_test, new_x_train, new_y_test, new_y_train = train_test_split(new_x, new_y, test_size=0.2, random_state=42)

print(f"Çoklu Lineer Regresyon Eğitim Seti Boyutu: {new_x_train.shape}")
print(f"Çoklu Lineer Regresyon Test Seti Boyutu: {new_x_test.shape}")

print("\nÇoklu Lineer Regresyon Modeli Eğitiliyor...")

model2 = LinearRegression()
model2.fit(new_x_train, new_y_train)
print("Çoklu Lineer Regresyon Modeli Eğitildi!\n")
print("Katsayısal:")
print(model2.coef_)
print(f"Intercept (b0) : {model2.intercept_}")

# r2 skoru
y_pred1 = model2.predict(new_x_test)
r2 = r2_score(new_y_test, y_pred1)
print(f"\nÇoklu Lineer Regresyon r2 skoru = {r2:.4f}")


# Hata Metrikleri
print("\nBasit ve Çoklu Lineer Regresyon Hata Metrikleri ")
mse = mean_squared_error(y_test, y_pred)
print(f"\nBasit Lineer Regresyon Mean Squared Error (MSE): {mse:.4f}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Basit Lineer Regresyon Mean Absolute Error (MAE): {mae:4f}")

mse1 = mean_squared_error(new_y_test, y_pred1)
print(f"Çoklu Lineer Regresyon Mean Squared Error (MSE): {mse1:.4f}")
mae1 = mean_absolute_error(new_y_test, y_pred1)
print(f"Çoklu Lineer Regresyon Mean Absolute Error (MAE): {mae1:4f}")
