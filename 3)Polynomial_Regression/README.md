## 3.Polynomial Regression

Polynomial Regression, bağımsız değişkenlerin polinomiyal bir kombinasyonunu kullanarak bağımlı değişkeni tahmin etmek için kullanılan bir tür regresyon analizidir.

Polynomial Regression modeli, aşağıdaki formda bir hipotez fonksiyonuna dayanır:

```
y = θ0 + θ1*x + θ2*x^2 + ... + θn*x^n

```

Burada:

- `y` hedef değişkeni temsil eder.
- `x` özelliği (bağımsız değişken) temsil eder.
- `θ0, θ1, ..., θn` modelin parametrelerini (veya ağırlıklarını) temsil eder.
- `n` polinomun derecesini temsil eder.

Modelin amacı, hedef değişkeni ve özelliğin polinomiyal kombinasyonu arasındaki ilişkiyi en iyi temsil eden parametreleri bulmaktır. Bu genellikle, gerçek hedef değerleri ve modelin tahminleri arasındaki kare hataların toplamını minimize ederek yapılır.

Polynomial Regression, `sklearn.preprocessing` modülündeki `PolynomialFeatures` sınıfı ve `sklearn.linear_model` modülündeki `LinearRegression` sınıfı ile Python'da kolayca uygulanabilir. `PolynomialFeatures` sınıfı, özelliklerin polinomiyal kombinasyonlarını oluşturur, ve `LinearRegression` sınıfı, bu polinomiyal özellikler üzerinde regresyon modelini eğitir.

Polynomial Regression, özellikle hedef değişken ve özellikler arasındaki ilişkinin doğrusal olmadığı durumlarda kullanışlıdır. 

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Örnek veri
X = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polinomiyal özellikleri oluştur
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Tahminler yap
y_pred = model.predict(X_test_poly)

# Hata hesapla
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', rmse)
```

Polynomial Regression'ın kullanılabileceği bazı örnek senaryolar şunlardır:

1. **Fiyat Tahmini:** Bir ürünün fiyatını, ürünün özelliklerinin polinomiyal kombinasyonlarına (örneğin, bir evin fiyatını, evin büyüklüğü ve yaşının karesine) göre tahmin etmek.
2. **Hava Durumu Tahmini:** Bir yerin sıcaklığını, zamanın polinomiyal kombinasyonlarına (örneğin, günün saati ve mevsimin karesine) göre tahmin etmek.
3. **Hastalık İlerlemesi Tahmini:** Bir hastanın hastalığının ilerlemesini, çeşitli sağlık göstergelerinin polinomiyal kombinasyonlarına (örneğin, yaş, kan basıncı, kolesterol seviyesi, vb.) göre tahmin etmek.

Bu senaryoların her birinde, Polynomial Regression modeli, bağımsız değişkenler ve hedef değişken arasındaki karmaşık ve doğrusal olmayan ilişkileri modellemek için kullanılır. Ancak, polinomun derecesi arttıkça, modelin karmaşıklığı da artar ve overfitting riski oluşabilir. Bu nedenle, polinomun derecesini belirlerken dikkatli olunmalıdır.
