### 6.**k-Nearest Neighbors (kNN) Classifier**

k-Nearest Neighbors (kNN) sınıflandırma algoritması, bir örneğin sınıfını, en yakın k komşusunun sınıflarına dayanarak tahmin eder. Bu, aşağıdaki adımları içerir:

1. Bir örneğin sınıfını tahmin etmek için, öncelikle örneğin en yakın k komşusunu buluruz. Bu genellikle öklidyen mesafe gibi bir mesafe ölçüsü kullanılarak yapılır.
2. Daha sonra, bu k komşunun en yaygın sınıfını buluruz. Bu, örneğin tahmini sınıfıdır.

kNN algoritması, aşağıdaki avantajlara sahiptir:

- Basit ve anlaşılır bir algoritmadır.
- Eğitim süresi yoktur çünkü tüm hesaplamalar tahmin aşamasında yapılır.
- Hem sınıflandırma hem de regresyon problemlarında kullanılabilir.

Ancak, kNN algoritması aşağıdaki dezavantajlara da sahiptir:

- Büyük veri setleri için tahmin süresi uzun olabilir çünkü her tahmin için tüm veri seti üzerinde bir mesafe hesaplaması yapılır.
- K değerinin seçimi, modelin performansını büyük ölçüde etkiler ve genellikle çapraz doğrulama gibi teknikler kullanılarak belirlenir.
- Özelliklerin ölçeklendirilmesi gereklidir çünkü kNN, mesafeye dayalı bir algoritmadır.

Python'da, `sklearn.neighbors` modülündeki `KNeighborsClassifier` sınıfı ile kNN sınıflandırma modeli oluşturulabilir ve eğitilebilir.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Iris veri setini yükle
data = load_iris()
X = data.data
y = data.target

# Veri setini eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Tahminler yap
y_pred = model.predict(X_test)

# Doğruluk hesapla
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)
```

k-Nearest Neighbors (kNN) sınıflandırma algoritmasının kullanılabileceği bazı örnek senaryolar şunlardır:

1. **Iris Çiçeği Sınıflandırması:** Iris çiçeğinin türünü, çiçeğin taç ve çanak yapraklarının boyutlarına göre tahmin etmek.
2. **El Yazısı Tanıma:** Bir el yazısı rakamının değerini, rakamın piksel değerlerine göre tahmin etmek.
3. **E-posta Spam Tespiti:** Bir e-postanın spam olup olmadığını, e-postanın içeriğine göre tahmin etmek.

Bu senaryoların her birinde, kNN sınıflandırma modeli, bağımsız değişkenler ve hedef değişken arasındaki ilişkiyi modellemek için kullanılır. Ancak, kNN modelinin performansı, k değerine ve kullanılan mesafe ölçüsüne bağlıdır, bu nedenle model seçimi ve hiperparametre ayarı önemlidir.
