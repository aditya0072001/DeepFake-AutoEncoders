from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

train_images_path = 'dataset/Modi'
train_cleaned_path = 'dataset/Trump'
train_images = sorted(os.listdir(train_images_path))
train_cleaned = sorted(os.listdir(train_cleaned_path))

X = []
y = []

for img in train_images:
    img_path = os.path.join(train_images_path, img)
    im = load_img(img_path,color_mode = 'grayscale',target_size = (540, 260))
    im = img_to_array(im).astype('float32')/255
    X.append(im)
for img in train_cleaned:
    img_path = os.path.join(train_cleaned_path, img)
    im = load_img(img_path,color_mode = 'grayscale', target_size = (540, 260))
    im = img_to_array(im).astype('float32')/255
    y.append(im)
    
X = np.array(X)
y = np.array(y)

print(X.shape,y.shape)

input_img = Input(shape=(540, 260, 1)) 

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)



x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')
history = autoencoder.fit(X, X, epochs = 1000, verbose = True)


input_img1 = Input(shape=(540, 260, 1))  

y = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img1)
y = MaxPooling2D((2, 2), padding='same')(y)
y = Conv2D(32, (3, 3), activation='relu', padding='same')(y)
y = MaxPooling2D((2, 2), padding='same')(y)
y = Conv2D(16, (3, 3), activation='relu', padding='same')(y)
encoded1 = MaxPooling2D((2, 2), padding='same')(y)


y = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded1)
y = UpSampling2D((2, 2))(y)
y = Conv2D(32, (3, 3), activation='relu', padding='same')(y)
y = UpSampling2D((2, 2))(y)
y = Conv2D(32, (3, 3), activation='relu')(y)
y = UpSampling2D((2, 2))(y)
decoded2 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(y)

autoencoder1 = Model(input_img1, decoded2)
autoencoder1.compile(optimizer='Adam', loss='binary_crossentropy')
history1 = autoencoder1.fit(y, y, epochs = 1000, verbose = True)

deepmodel = Model(encoded,decoded2)

deepmodel.save('deepmodel.h5')
