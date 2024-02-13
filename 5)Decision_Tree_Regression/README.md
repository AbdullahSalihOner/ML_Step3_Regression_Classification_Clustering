### 5.Decision Tree Regression

Decision Tree Regression, karar ağaçları algoritmasının regresyon problemlarına uygulanmış halidir. Bu algoritma, bağımsız değişkenlerin bir dizi karar kuralına dayalı olarak bağımlı değişkeni tahmin etmek için kullanılır.

Karar ağacı, bir dizi karar kuralına dayalı bir ağaç yapısı oluşturur. Her düğümde, bir özelliğin belirli bir eşiği ile karşılaştırılması sonucunda bir karar verilir. Bu süreç, bir yaprak düğümüne ulaşana kadar devam eder. Yaprak düğümünde, hedef değişkenin tahmini değeri bulunur.

Decision Tree Regression modeli, aşağıdaki avantajlara sahiptir:

- Hem kategorik hem de sürekli özelliklerle çalışabilir.
- Verilerin ölçeklendirilmesi veya normalleştirilmesi gerekmez.
- Modelin çıktısı, karar ağacının yapısı nedeniyle yorumlanması kolaydır.

Ancak, Decision Tree Regression modelinin bazı dezavantajları da vardır:

- Karar ağaçları, verilere aşırı uyum sağlama (overfitting) eğilimindedir. Bu, ağacın derinliğini sınırlayarak veya ağacı budayarak (pruning) hafifletilebilir.
- Karar ağaçları, verilerdeki küçük değişikliklere karşı duyarlıdır. Verilerdeki küçük bir değişiklik, ağacın yapısını büyük ölçüde değiştirebilir.
- Karar ağaçları, doğrusal olmayan ve özellikler arasında etkileşim olan ilişkileri modelleyebilir, ancak doğrusal ilişkileri modellemekte zorlanabilir.

Python'da, `sklearn.tree` modülündeki `DecisionTreeRegressor` sınıfı ile Decision Tree Regression modeli oluşturulabilir ve eğitilebilir.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
import numpy as np

# Boston ev fiyatları veri setini yükle
data = load_boston()
X = data.data
y = data.target

# Veri setini eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = DecisionTreeRegressor(max_depth=3)
model.fit(X_train, y_train)

# Tahminler yap
y_pred = model.predict(X_test)

# Hata hesapla
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', rmse)
```

Decision Tree Regression'ın kullanılabileceği bazı örnek senaryolar şunlardır:

1. **Ev Fiyatı Tahmini:** Bir evin fiyatını, evin çeşitli özelliklerine (örneğin, büyüklüğü, konumu, yaş, vb.) göre tahmin etmek.
2. **Enerji Tüketimi Tahmini:** Bir bina veya cihazın enerji tüketimini, çeşitli faktörlere (örneğin, kullanım süresi, hava durumu, bina özellikleri, vb.) göre tahmin etmek.
3. **Sağlık Tahminleri:** Bir hastanın sağlık durumunu, çeşitli sağlık göstergelerine (örneğin, yaş, kan basıncı, kolesterol seviyesi, vb.) göre tahmin etmek.

Bu senaryoların her birinde, Decision Tree Regression modeli, bağımsız değişkenler ve hedef değişken arasındaki ilişkiyi modellemek için kullanılır. Ancak, karar ağaçları, verilere aşırı uyum sağlama (overfitting) eğilimindedir, bu nedenle modelin karmaşıklığını kontrol etmek için ağacın derinliğini sınırlamak önemlidir.
