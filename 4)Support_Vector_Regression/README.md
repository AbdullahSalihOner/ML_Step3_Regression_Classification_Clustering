### 4.Support Vector Regression

Support Vector Regression (SVR), Support Vector Machines (SVM) algoritmasının regresyon problemlarına uygulanmış halidir. SVM, genellikle sınıflandırma problemları için kullanılırken, SVR sürekli hedef değişkenlerin tahmin edilmesi için kullanılır.

SVR, bir marj hata içinde en küçük hata ile modeli eğitmeye çalışır. Marj hata, genellikle epsilon (ε) ile belirlenir. SVR, bu epsilon-insensitive loss function ile, hedef değerler ile tahminler arasındaki farkın epsilon'dan büyük olduğu durumlarda hata hesaplar.

SVR, doğrusal (Linear SVR) ve doğrusal olmayan (Kernel SVR) olmak üzere iki türdür. Doğrusal SVR, doğrusal bir çizgi (1 boyut), düzlem (2 boyut) veya hiper düzlem (daha yüksek boyutlar) kullanarak verileri ayırır. Kernel SVR, doğrusal olmayan bir çizgi, düzlem veya hiper düzlem kullanarak verileri ayırır. Bu, kernel trick adı verilen bir teknik kullanılarak yapılır.

SVR, `sklearn.svm` modülündeki `SVR` sınıfı ile Python'da kolayca uygulanabilir. Bu sınıf, modeli eğitmek için `fit` metodunu ve yeni veriler üzerinde tahminler yapmak için `predict` metodunu sağlar.

SVR, özellikle hedef değişken ve özellikler arasındaki ilişkinin doğrusal olmadığı ve/veya verilerin yüksek boyutlu olduğu durumlarda kullanışlıdır. 

```python
from sklearn.svm import SVR
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
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
model.fit(X_train, y_train)

# Tahminler yap
y_pred = model.predict(X_test)

# Hata hesapla
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', rmse)
```

Support Vector Regression (SVR), çeşitli gerçek dünya problemlerinin çözümünde kullanılabilir. İşte bazı örnek kullanım senaryoları:

1. **Fiyat Tahmini:** Bir ürünün fiyatını, ürünün çeşitli özelliklerine (örneğin, bir evin fiyatını, evin büyüklüğü, konumu, yaş, vb.) göre tahmin etmek.
2. **Hisse Senedi Fiyatı Tahmini:** Bir hisse senedinin fiyatını, çeşitli ekonomik göstergelere (örneğin, faiz oranları, işsizlik oranı, vb.) göre tahmin etmek.
3. **Enerji Tüketimi Tahmini:** Bir bina veya cihazın enerji tüketimini, çeşitli faktörlere (örneğin, kullanım süresi, hava durumu, bina özellikleri, vb.) göre tahmin etmek.
4. **Sağlık Tahminleri:** Bir hastanın sağlık durumunu, çeşitli sağlık göstergelerine (örneğin, yaş, kan basıncı, kolesterol seviyesi, vb.) göre tahmin etmek.

Bu senaryoların her birinde, SVR modeli, bağımsız değişkenler ve hedef değişken arasındaki karmaşık ve doğrusal olmayan ilişkileri modellemek için kullanılır. Ancak, SVR modelinin performansı, kullanılan kernel fonksiyonu ve parametrelerine bağlıdır, bu nedenle model seçimi ve hiperparametre ayarı önemlidir.
