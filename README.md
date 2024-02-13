<h1>MachineLearning_Step3_Regression_Classification_Clustering</h1>

### SUPERVİSED LEARNING

Supervised Learning (Denetimli Öğrenme), bir makine öğrenmesi yaklaşımıdır. Bu yaklaşımda, bir model, giriş verileri (features) ve bu verilere karşılık gelen çıktılar (labels) arasındaki ilişkiyi öğrenmek için eğitilir. Model, bu ilişkiyi öğrendikten sonra, çıktıları bilinmeyen yeni giriş verileri için tahminler yapabilir.

Supervised Learning'in iki ana türü vardır: Classification (Sınıflandırma) ve Regression (Regresyon). Sınıflandırma, çıktının belirli sınıflardan birine ait olduğu durumlar için kullanılır (örneğin, bir e-postanın spam olup olmadığını belirlemek). Regresyon, çıktının sürekli bir değer olduğu durumlar için kullanılır (örneğin, bir evin fiyatını tahmin etmek).

Supervised Learning'in avantajları şunlardır:

- Modelin performansını ölçmek kolaydır, çünkü gerçek çıktılar bilinir.
- Çeşitli uygulamalar için kullanılabilir, örneğin spam filtreleme, görüntü sınıflandırma, fiyat tahmini vb.

Ancak, Supervised Learning'in bazı dezavantajları da vardır:

- Etiketlenmiş veri gerektirir, bu da etiketleme maliyeti olabilir.
- Model, eğitim verilerinde görünmeyen yeni durumları genellikle iyi bir şekilde işleyemez (overfitting).

### UNSUPERVISED LEARNING

Unsupervised Learning (Denetimsiz Öğrenme), bir makine öğrenmesi yaklaşımıdır. Bu yaklaşımda, bir model, giriş verileri (features) arasındaki yapıyı veya ilişkileri öğrenmek için eğitilir. Bu tür bir öğrenme, çıktı verileri (labels) olmadan gerçekleşir, bu yüzden model, verilerin içindeki gizli desenleri veya yapıları bulmaya çalışır.

Unsupervised Learning'in iki ana türü vardır: Clustering (Kümeleme) ve Dimensionality Reduction (Boyut Azaltma). Kümeleme, veri noktalarını benzerliklerine göre gruplara ayırır. Boyut Azaltma, verinin boyutunu (özellik sayısını) azaltırken verinin yapısını korumaya çalışır.

Unsupervised Learning'in avantajları şunlardır:

- Etiketlenmiş veri gerektirmez, bu yüzden etiketleme maliyeti yoktur.
- Veri setindeki gizli desenleri veya yapıları bulabilir.
- Veri boyutunu azaltabilir, bu da hesaplama maliyetini azaltır ve veriyi görselleştirmeyi kolaylaştırır.

Ancak, Unsupervised Learning'in bazı dezavantajları da vardır:

- Modelin performansını ölçmek zordur, çünkü gerçek çıktılar genellikle bilinmez.
- Modelin bulduğu desenler veya yapılar, insanlar için anlamlı veya kullanışlı olmayabilir.


### Overfitting

Overfitting, makine öğrenmesi modellerinde karşılaşılan yaygın bir problem olup, modelin eğitim verilerini aşırı öğrendiği ve bu nedenle yeni verilere genelleme yapma yeteneğinin azaldığı durumu ifade eder.

Gürültülü Veri (Noisy Data): **Veri girişi veya veri toplanması esnasında oluşan sistem dışı hatalara gürültülü denir**.

Overfitting durumunda, model eğitim verilerindeki gürültüyü ve ayrıntıları öğrenir ve bu, modelin eğitim verilerinde yüksek performans göstermesine neden olur. Ancak, model test verileri veya yeni veriler üzerinde test edildiğinde, performansı genellikle düşer. Bu, modelin eğitim verilerindeki özel ayrıntıları ve gürültüyü öğrendiği ve bu nedenle yeni verilere genelleme yapma yeteneğinin azaldığı anlamına gelir. Yani eğitim verilerini ezberler. Overfitting şu durumlarda ortaya çıkabilir:

- **Model Çok Karmaşık**: Eğer model çok fazla parametreye sahipse veya veri setine göre aşırı karmaşıksa, eğitim verilerindeki gürültüyü veya hataları öğrenmeye başlayabilir.
- **Yetersiz Eğitim Verisi**: Eğitim verisi yetersiz olduğunda model, verilerdeki özel özellikleri öğrenir ve genelleme yapamaz hale gelir.
- **Uzun Eğitim Süreleri**: Aşırı eğitim (overtraining), özellikle derin öğrenme modellerinde overfitting'e yol açabilir.

Overfitting'i önlemek için çeşitli teknikler kullanılabilir:

1. **Veri setini genişletme:** Daha fazla veri toplamak ve modeli daha çeşitli verilerle eğitmek, modelin genelleme yeteneğini artırabilir.
2. **Regularizasyon:** Regularizasyon teknikleri, modelin karmaşıklığını sınırlayarak overfitting'i önlemeye yardımcı olabilir. Örneğin, L1 ve L2 regularizasyonu gibi teknikler, modelin ağırlıklarını sınırlar ve bu da modelin eğitim verilerini aşırı öğrenmesini önler. Modelin ağırlıklarını sınırlamak ve karmaşıklığını azaltmak.
3. **Cross-validation:** Cross-validation, modelin farklı veri alt kümeleri üzerinde performansını değerlendirmek için kullanılır. Bu, modelin genelleme yeteneğini artırır ve overfitting'i önler.
4. **Early stopping:** Eğitim sürecinde, eğer modelin doğrulama veri seti üzerindeki performansı düşmeye başlarsa, eğitim sürecini durdurmak ve modelin aşırı öğrenmesini önlemek için early stopping kullanılabilir.
5. **Model karmaşıklığını azaltma:** Daha basit bir model kullanmak veya modelin parametrelerini sınırlamak (örneğin, bir karar ağacının derinliğini sınırlamak), modelin eğitim verilerini aşırı öğrenmesini önleyebilir.

### METRICS ;

Makine öğrenmesi modellerinin performansını değerlendirmek için çeşitli metrikler kullanılır. Bu metrikler, modelin ne kadar iyi tahminler yaptığını ölçer. İşte bazı yaygın metrikler:

1. **Doğruluk (Accuracy):** Doğru tahminlerin toplam tahminlere oranıdır. Genellikle sınıflandırma problemleri için kullanılır.
2. **Hata Oranı (Error Rate):** Yanlış tahminlerin toplam tahminlere oranıdır. Doğruluk metriğinin tam tersidir.
3. **Precision (Kesinlik):** Pozitif olarak tahmin edilen örneklerin ne kadarının gerçekten pozitif olduğunu ölçer.
4. **Recall (Duyarlılık):** Gerçek pozitif örneklerin ne kadarının doğru bir şekilde tahmin edildiğini ölçer.
5. **F1 Score:** Precision ve Recall'ın harmonik ortalamasıdır. Hem Precision hem de Recall'ı dikkate alır.
6. **ROC AUC:** Receiver Operating Characteristic (ROC) eğrisinin altında kalan alanın (Area Under Curve - AUC) ölçüsüdür. Sınıflandırma modelinin performansını değerlendirmek için kullanılır.
7. **Mean Absolute Error (MAE):** Gerçek değerler ile tahminler arasındaki mutlak hataların ortalamasıdır. Regresyon problemleri için kullanılır.
8. **Mean Squared Error (MSE):** Gerçek değerler ile tahminler arasındaki kare hataların ortalamasıdır. Regresyon problemleri için kullanılır.
9. **Root Mean Squared Error (RMSE):** MSE'nin kareköküdür. Regresyon problemleri için kullanılır. RMSE değeri 0'a ne kadar yakınsa model o kadar iyi demektir.
10. **R-squared (R^2):** Bağımlı değişkendeki varyansın ne kadarının bağımsız değişkenler tarafından açıklandığını ölçer. Regresyon problemleri için kullanılır. R^2 değeri 1'e ne kadar yakınsa model o kadar iyi demektir.

Bu metrikler, modelin performansını değerlendirmek ve farklı modelleri karşılaştırmak için kullanılır. Hangi metriğin kullanılacağı, problemin türüne ve iş gereksinimlerine bağlıdır. Biz uygulamalarımız da çoğunlukla RMSE ve R^2 değerlerini kullanacağız.

- Model oluşturma ve tahmin yapma süreci örnek adımları ;

```python
# 1) Modeli Oluştur
lr = LinearRegression()  # Lineer regresyon modelini oluşturduk

# 2) Modeli Eğit (x_train ve y_train kullanarak)
lr.fit(x_train, y_train)  # Modeli eğitim verileriyle eğittik

# 3) Modeli Test Et (x_test kullanarak)
y_head = lr.predict(x_test)  # Test verileri üzerinde modeli kullanarak tahmin yaptık
```

### 1.**Linear Regression (Basit Doğrusal Regresyon)**


### 2.**Multiple Linear Regression (Çoklu Doğrusal Regresyon)**


## 3.Polynomial Regression


### 4.Support Vector Regression


### 5.Decision Tree Regression


### 6.**k-Nearest Neighbors (kNN) Classifier**


### 7.Logistic Regression


