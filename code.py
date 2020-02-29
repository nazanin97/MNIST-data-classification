import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


# load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
print(x_train.shape)


# first part
# Reshaping the array and Normalizing the RGB codes by dividing it to the max RGB value
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# set an optimizer with a given loss function which uses a metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fit the model by using our train data
t1 = model.fit(x_train, y_train, epochs=10)


# Display the model's architecture
model.summary()

# evaluate the trained model with x_test and y_test
loss, accuracy = model.evaluate(x_test, y_test)


# Second part
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

model2 = Sequential()

model2.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model2.add(MaxPooling2D(pool_size=(2, 2)))
# Flattening the 2D arrays for fully connected layers
model2.add(Flatten())
model2.add(Dense(128, activation=tf.nn.relu))
model2.add(Dropout(0.2))
model2.add(Dense(10, activation=tf.nn.softmax))


# set an optimizer with a given loss function which uses a metric
model2.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# fit the model by using our train data
t2 = model2.fit(x=x_train, y=y_train, epochs=10)


# Display the model's architecture
model2.summary()


# evaluate the trained model with x_test and y_test
loss, accuracy = model2.evaluate(x_test, y_test)
