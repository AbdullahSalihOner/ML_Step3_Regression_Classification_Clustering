### 2.**Multiple Linear Regression (Çoklu Doğrusal Regresyon)**

Çoklu Linear Regresyon, birden fazla bağımsız değişken (özellik) ve bir bağımlı değişken (hedef) arasındaki doğrusal ilişkiyi modellemek için kullanılan bir denetimli öğrenme algoritmasıdır.

Çoklu Linear Regresyon modeli, aşağıdaki formda bir hipotez fonksiyonuna dayanır:

```
y = θ0 + θ1*x1 + θ2*x2 + ... + θn*xn
```

Burada:

- `y` hedef değişkeni temsil eder.
- `x1, x2, ..., xn` özellikleri (bağımsız değişkenler) temsil eder.
- `θ0, θ1, ..., θn` modelin parametrelerini (veya ağırlıklarını) temsil eder.

Modelin amacı, hedef değişkeni ve özellikler arasındaki ilişkiyi en iyi temsil eden parametreleri bulmaktır. Bu genellikle, gerçek hedef değerleri ve modelin tahminleri arasındaki kare hataların toplamını minimize ederek yapılır.

Çoklu Linear Regresyon, `sklearn.linear_model` modülündeki `LinearRegression` sınıfı ile Python'da kolayca uygulanabilir. Bu sınıf, modeli eğitmek için `fit` metodunu ve yeni veriler üzerinde tahminler yapmak için `predict` metodunu sağlar.

Çoklu Linear Regresyon, özellikle birden çok özelliğin hedef değişken üzerindeki etkisini anlamak ve bu ilişkileri kullanarak hedef değişkenin değerlerini tahmin etmek için kullanışlıdır. Örnek bir uygulama;

```python
from sklearn.linear_model import LinearRegression
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
model = LinearRegression()
model.fit(X_train, y_train)

# Tahminler yap
y_pred = model.predict(X_test)

# Hata hesapla
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', rmse)
```

Multiple Linear Regression'ın kullanılabileceği bazı örnek senaryolar şunlardır:

1. **Ev Fiyatı Tahmini:** Bir evin fiyatını, evin çeşitli özelliklerine (örneğin, büyüklük, konum, yaş, vb.) göre tahmin etmek.
2. **Hisse Senedi Fiyatı Tahmini:** Bir hisse senedinin fiyatını, çeşitli ekonomik göstergelere (örneğin, faiz oranları, işsizlik oranı, vb.) göre tahmin etmek.
3. **Enerji Tüketimi Tahmini:** Bir bina veya cihazın enerji tüketimini, çeşitli faktörlere (örneğin, kullanım süresi, hava durumu, bina özellikleri, vb.) göre tahmin etmek.
4. **Satış Tahmini:** Bir ürünün satış miktarını, çeşitli faktörlere (örneğin, fiyat, reklam harcamaları, mevsimsel trendler, vb.) göre tahmin etmek.

Bu senaryoların her birinde, Multiple Linear Regression modeli, bağımsız değişkenler ve hedef değişken arasındaki ilişkiyi modellemek için kullanılır. Ancak, özellikler arasında çoklu doğrusallık (multicollinearity) gibi sorunlar oluşabilir, bu da modelin tahminlerini etkileyebilir. Bu tür sorunları önlemek için, özellik seçimi veya düzenlileştirme (regularization) gibi teknikler kullanılabilir.

- Multicollinearity (Çoklu Doğrusallık);

Multicollinearity (Çoklu Doğrusallık), regresyon analizinde karşılaşılan bir durum olup, bağımsız değişkenlerin birbirleriyle yüksek oranda korelasyonlu olduğu durumu ifade eder. Yani, bir özelliğin değerleri, diğer bir veya daha fazla özelliğin değerlerini tahmin etmek için kullanılabilir.

Multicollinearity, regresyon katsayılarının tahminlerini etkileyebilir ve bu da modelin kararlılığını ve güvenilirliğini azaltabilir. Özellikle, çoklu doğrusallık durumunda, regresyon katsayılarının tahminleri genellikle büyük standart hatalara sahip olur, bu da katsayıların istatistiksel olarak anlamlı olmamasına neden olabilir.

Multicollinearity'yi tespit etmek için çeşitli yöntemler vardır. Bunlardan biri, Varyans Enflasyon Faktörü (VIF) hesaplamaktır. VIF, bir özelliğin diğer özellikler tarafından ne kadar iyi tahmin edilebildiğinin bir ölçüsüdür. VIF değeri 1'den çok daha büyük olan bir özellik, çoklu doğrusallık sorunu olduğuna işaret edebilir.

Multicollinearity'yi önlemek veya azaltmak için çeşitli stratejiler vardır. Bunlar arasında özellik seçimi, özellik mühendisliği ve düzenlileştirme (regularization) tekniklerini kullanmak bulunur. Bu teknikler, modelin genelleme yeteneğini artırabilir ve overfitting riskini azaltabilir.

- Düzenlileştirme (Regularization),

Düzenlileştirme (Regularization), makine öğrenmesi modellerinin aşırı uyumunu (overfitting) önlemek için kullanılan bir tekniktir. Düzenlileştirme, modelin karmaşıklığını sınırlayarak ve/veya modelin ağırlıklarını küçültmek suretiyle çalışır. Bu, modelin eğitim verilerini aşırı öğrenmesini önler ve genelleme yeteneğini artırır.

Düzenlileştirme teknikleri genellikle iki ana türdür: L1 Düzenlileştirme (Lasso) ve L2 Düzenlileştirme (Ridge).

1. **L1 Düzenlileştirme (Lasso):** L1 düzenlileştirme, modelin hata fonksiyonuna ağırlıkların mutlak değerlerinin toplamını ekler. Bu, bazı ağırlıkları tamamen sıfıra indirger, bu da modelin daha az özellik kullanmasına ve daha "seyrek" bir model oluşturmasına neden olur.
2. **L2 Düzenlileştirme (Ridge):** L2 düzenlileştirme, modelin hata fonksiyonuna ağırlıkların karelerinin toplamını ekler. Bu, ağırlıkları sıfıra indirgemez, ancak onları küçültür, bu da modelin daha düzgün ve daha az karmaşık olmasını sağlar.

Düzenlileştirme teknikleri, `sklearn.linear_model` modülündeki `Lasso` ve `Ridge` sınıfları ile Python'da kolayca uygulanabilir. Bu sınıflar, modeli eğitmek için `fit` metodunu ve yeni veriler üzerinde tahminler yapmak için `predict` metodunu sağlar.

Düzenlileştirme, özellikle çok sayıda özelliğin olduğu ve/veya özellikler arasında çoklu doğrusallık (multicollinearity) gibi sorunların olduğu durumlarda kullanışlıdır. Bu teknikler, modelin genelleme yeteneğini artırabilir ve overfitting riskini azaltabilir.

 `sklearn` kütüphanesinin `Lasso` ve `Ridge` sınıflarını kullanarak L1 ve L2 düzenlileştirmeyi nasıl uygulanır ;

```python
from sklearn.linear_model import Lasso, Ridge
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

# Lasso (L1 Düzenlileştirme) modelini oluştur ve eğit
lasso = Lasso(alpha=0.1)  # alpha parametresi düzenlileştirme gücünü kontrol eder
lasso.fit(X_train, y_train)

# Ridge (L2 Düzenlileştirme) modelini oluştur ve eğit
ridge = Ridge(alpha=0.1)  # alpha parametresi düzenlileştirme gücünü kontrol eder
ridge.fit(X_train, y_train)

# Her iki modelin tahminlerini al ve hata hesapla
lasso_preds = lasso.predict(X_test)
ridge_preds = ridge.predict(X_test)

lasso_error = np.sqrt(mean_squared_error(y_test, lasso_preds))
ridge_error = np.sqrt(mean_squared_error(y_test, ridge_preds))

print("Lasso RMSE: ", lasso_error)
print("Ridge RMSE: ", ridge_error)
```

Bu örnekte, Boston ev fiyatları veri setini kullanıyoruz. Veri setini eğitim ve test setlerine ayırıyoruz. Ardından, hem Lasso (L1) hem de Ridge (L2) düzenlileştirmeyi uygulayan iki model oluşturuyoruz. Her iki modeli de eğitim verileriyle eğitiyoruz ve test verileri üzerinde tahminler yapıyoruz. Son olarak, her iki modelin performansını, gerçek hedef değerleri ve modelin tahminleri arasındaki karekök ortalama kare hata (RMSE) ile değerlendiriyoruz.
