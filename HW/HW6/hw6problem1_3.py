

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


plt.imshow(x_train[0])



x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255



(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'valid set')
print(x_test.shape[0], 'test set')



model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 2,
                                padding = 'same', activation = 'relu', input_shape = (28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size = 2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 2,
                                padding = 'same', activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = 2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

model.summary()



model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])


checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only = True)


#%%

model.fit(x_train,
          y_train,
          batch_size = 64,
          epochs = 10,
          validation_data = (x_valid, y_valid),
          callbacks = [checkpointer])

#%%

model.load_weights('model.weights.best.hdf5')

#%%

score = model.evaluate(x_test, y_test, verbose = 0)
print('\n', 'Test accuracy:', score[1])



#
# y_hat = model.predict(x_test)
#
# figure = plt.figure(figsize = (20,8))
# for i, index in enumerate(np.random.choice(x_test.shape[0], size = 15, replace = False)):
#     ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
#     ax.imshow(np.squeeze(x_test[index]))
#     predict_index = np.argmax(y_hat[index])
#     true_index = np.argmax(y_test[index])
#     ax.set_title("{} ({})".format(class_names[predict_index],
#                                   class_names[true_index]),
#                 color = ('green' if predict_index == true_index else 'red'))
#
# #%%


