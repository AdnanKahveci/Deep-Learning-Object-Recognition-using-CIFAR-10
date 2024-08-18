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


#Kullanım
##Veri Setini Yükleme ve Ön İşleme:
###CIFAR-10 veri seti otomatik olarak indirilir ve normalize edilir.

####Veriyi yüklemek ve normalize etmek için:
```bash
import tensorflow as tf
from tensorflow.keras import datasets

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train = X_train / 255
X_test = X_test / 255

####Modelin Oluşturulması:
####CNN mimarisi oluşturulur ve modelin katmanları tanımlanır:

####İlk Convolutional katmanlar ve MaxPooling2D katmanları ile özellikler çıkarılır.
####Flatten ve Dense katmanları ile model eğitilir.

```bash
from tensorflow.keras import layers, models

deep_learning_model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
