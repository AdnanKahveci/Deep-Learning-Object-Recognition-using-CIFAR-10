# Deep-Learning-Object-Recognition-using-CIFAR-10
Object Classification with Deep Learning using CIFAR-10 Dataset
# CIFAR-10 Object Classification with CNN

## Proje Açıklaması
Bu proje, CIFAR-10 veri setini kullanarak nesne tanıma ve sınıflandırma işlemi gerçekleştiren bir derin öğrenme modelini içerir. Projede, Convolutional Neural Network (CNN) kullanılarak 32x32 piksel boyutunda renkli fotoğrafların 10 farklı sınıftan birine ait olup olmadığı belirlenir.

## Veri Seti
CIFAR-10 veri seti, 10 farklı sınıfa ait 60.000 32x32 piksellik renkli görüntü içerir. Bu sınıflar şunlardır:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Gereksinimler
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

Bu kütüphaneleri yüklemek için aşağıdaki komutları kullanabilirsiniz:
```bash
pip install tensorflow numpy matplotlib
```
## Kullanım
- Veriyi Yükleyin ve Ön İşleyin
  Veri setini yükleyip normalize ederek modelin eğitimine hazır hale getirin:
  ```bash
  import tensorflow as tf
  from tensorflow.keras import datasets
  
  (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
  X_train = X_train / 255
  X_test = X_test / 255
  ```
-Modeli Oluşturun
  CNN mimarisi oluşturun ve modelin katmanlarını tanımlayın:
  ```bash
    from tensorflow.keras import layers, models
  
    deep_learning_model = models.Sequential([
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
  ```
- Modeli Eğitin
  Modeli eğitmek için:
  ```bash
    deep_learning_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    deep_learning_model.fit(X_train, y_train, epochs=5)
  ```
- Modeli Değerlendirin ve Tahmin Yapın
  Modeli test veri seti üzerinde değerlendirin ve tahminler yapın:
  ```bash
    deep_learning_model.evaluate(X_test, y_test)
    y_pred = deep_learning_model.predict(X_test)
  ```
- Sonuçları Görselleştirin
  Test setinden örnekleri ve modelin tahminlerini görselleştirin:
  ```bash
    import matplotlib.pyplot as plt
    import numpy as np
    
    resim_siniflari = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    def plot_sample(X, y, index):
        plt.figure(figsize=(15, 2))
        plt.imshow(X[index])
        plt.xlabel(resim_siniflari[y[index]])
        plt.show()
    
    plot_sample(X_test, y_test, 0)
    plot_sample(X_test, y_test, 1)
    plot_sample(X_test, y_test, 2)
  ```
Bu proje [MIT Lisansı](https://opensource.org/licenses/MIT) altında lisanslanmıştır.
