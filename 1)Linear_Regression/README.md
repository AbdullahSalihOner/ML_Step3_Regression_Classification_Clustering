### 1.**Linear Regression (Basit Doğrusal Regresyon)**

Basit Linear Regresyon, bir bağımsız değişken (özellik) ve bir bağımlı değişken (hedef) arasındaki doğrusal ilişkiyi modellemek için kullanılan bir denetimli öğrenme algoritmasıdır.

Basit Linear Regresyon modeli, aşağıdaki formda bir hipotez fonksiyonuna dayanır:

```
y = θ0 + θ1*x

```

Burada:

- `y` hedef değişkeni temsil eder.
- `x` özelliği (bağımsız değişken) temsil eder.
- `θ0` ve `θ1` modelin parametrelerini (veya ağırlıklarını) temsil eder.

Modelin amacı, hedef değişkeni ve özelliği arasındaki ilişkiyi en iyi temsil eden parametreleri bulmaktır. Bu genellikle, gerçek hedef değerleri ve modelin tahminleri arasındaki kare hataların toplamını minimize ederek yapılır.

Basit Linear Regresyon, `sklearn.linear_model` modülündeki `LinearRegression` sınıfı ile Python'da kolayca uygulanabilir. Bu sınıf, modeli eğitmek için `fit` metodunu ve yeni veriler üzerinde tahminler yapmak için `predict` metodunu sağlar.

Basit Linear Regresyon, özellikle iki değişken arasındaki doğrusal ilişkiyi anlamak ve bu ilişkiyi kullanarak hedef değişkenin değerlerini tahmin etmek için kullanışlıdır. Ancak, gerçek dünya verileri genellikle karmaşıktır ve birden çok özelliği içerir, bu nedenle Çoklu Linear Regresyon gibi daha karmaşık modeller genellikle daha iyi performans gösterir.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Örnek veri
X = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Tahminler yap
y_pred = model.predict(X_test)

# Hata hesapla
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', rmse)
```

Basit Linear Regression'ın kullanılabileceği bazı örnek senaryolar şunlardır:

1. **Fiyat Tahmini:** Bir ürünün fiyatını, ürünün özelliklerine (örneğin, bir evin fiyatını, evin büyüklüğüne göre) tahmin etmek.
2. **Satış Tahmini:** Bir ürünün satış miktarını, reklam harcamalarına göre tahmin etmek.
3. **Hizmet Süresi Tahmini:** Bir hizmetin tamamlanma süresini, işin büyüklüğüne göre tahmin etmek.
4. **Enerji Tüketimi Tahmini:** Bir bina veya cihazın enerji tüketimini, kullanım süresine veya hava durumuna göre tahmin etmek.

Bu senaryoların her birinde, Basit Linear Regression modeli, bağımsız değişken ve hedef değişken arasındaki doğrusal ilişkiyi modellemek için kullanılır. Ancak, gerçek dünya verileri genellikle karmaşıktır ve birden çok özelliği içerir, bu nedenle Çoklu Linear Regression veya diğer daha karmaşık modeller genellikle daha iyi performans gösterir.
