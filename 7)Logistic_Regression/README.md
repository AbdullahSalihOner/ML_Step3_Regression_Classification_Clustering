### 7.Logistic Regression

Logistic Regression, sınıflandırma problemlarında kullanılan bir makine öğrenmesi algoritmasıdır. Özellikle ikili sınıflandırma problemlarında (yani, hedef değişkenin iki olası değeri olduğunda) yaygın olarak kullanılır.

Logistic Regression, bağımlı değişkenin olasılığını tahmin etmek için bir lojistik fonksiyonu kullanır. Bu, bağımsız değişkenlerin bir lineer kombinasyonunu alır ve bu değeri bir lojistik fonksiyonuna (veya sigmoid fonksiyonuna) geçirir. Çıktı, 0 ile 1 arasında bir değerdir ve bu, bağımlı değişkenin olasılığını temsil eder.

Logistic Regression modeli, aşağıdaki avantajlara sahiptir:

- Çıktı, bir olasılık olarak yorumlanabilir, bu da modelin tahminlerinin belirsizliğini ölçmeyi sağlar.
- Özellikler arasındaki etkileşimleri ve etkileşimlerin hedef değişken üzerindeki etkisini modelleyebilir.
- Regularization teknikleri (L1 ve L2) ile overfittingi önleyebilir.

Ancak, Logistic Regression modelinin bazı dezavantajları da vardır:

- Doğrusal bir sınıflandırıcıdır ve bu nedenle doğrusal olarak ayrılabilir olmayan verilerle iyi çalışmaz.
- Özelliklerin ölçeklendirilmesi gereklidir çünkü Logistic Regression, özelliklerin büyüklüğüne duyarlıdır.
- Outlierlara karşı duyarlıdır ve bu, modelin performansını etkileyebilir.

Python'da, `sklearn.linear_model` modülündeki `LogisticRegression` sınıfı ile Logistic Regression modeli oluşturulabilir ve eğitilebilir.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# Meme kanseri veri setini yükle
data = load_breast_cancer()
X = data.data
y = data.target

# Veri setini eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Tahminler yap
y_pred = model.predict(X_test)

# Doğruluk hesapla
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)
```

Logistic Regression'ın kullanılabileceği bazı örnek senaryolar şunlardır:

1. **Hastalık Teşhisi:** Bir hastanın belirli bir hastalığa sahip olup olmadığını, çeşitli sağlık göstergelerine (örneğin, yaş, kan basıncı, kolesterol seviyesi, vb.) göre tahmin etmek.
2. **Spam Tespiti:** Bir e-postanın spam olup olmadığını, e-postanın içeriğine göre tahmin etmek.
3. **Müşteri Kaybı Tahmini:** Bir müşterinin bir hizmeti bırakıp bırakmayacağını, müşterinin çeşitli özelliklerine (örneğin, yaş, cinsiyet, kullanım süresi, vb.) göre tahmin etmek.

Bu senaryoların her birinde, Logistic Regression modeli, bağımsız değişkenler ve hedef değişken arasındaki ilişkiyi modellemek için kullanılır. Ancak, Logistic Regression, doğrusal bir sınıflandırıcı olduğu için, doğrusal olarak ayrılabilir olmayan verilerle iyi çalışmayabilir. Bu durumda, kernel trick veya nonlinear dönüşümler gibi teknikler kullanılabilir.
