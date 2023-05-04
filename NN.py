import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

# load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape the data to be in the form of [samples][width][height][channels]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# normalize the pixel values from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one-hot encode the target variable
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define the model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))






# create a window
root = tk.Tk()
root.title("Drawing Canvas")

pixel_array = np.zeros((28, 28))
# create a canvas
canvas_width = 280
canvas_height = 280
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

# define event handlers
def paint(event):
    # get the mouse position
    x = event.x // 10 
    y = event.y // 10 
    pixel_array[y][x]=1
    # draw a rectangle on the canvas
    canvas.create_rectangle(x * 10, y * 10, x * 10 + 10, y * 10 + 10, fill='black')

def clear_canvas():
    global pixel_array
    pixel_array = np.zeros((28, 28))
    canvas.delete('all')

def quess_number():
    input_data = pixel_array.reshape(1, 28, 28, 1)
    predictions = model.predict(input_data)
    messagebox.showinfo(title="Prediction", message=str(np.argmax(predictions[0])))
# bind the event handlers
canvas.bind('<B1-Motion>', paint)

# create a button to clear the canvas
clear_button = tk.Button(root, text='Clear Canvas', command=clear_canvas)
quess_button = tk.Button(root, text='Quess', command=quess_number)
clear_button.pack()
quess_button.pack()

# start the event loop
root.mainloop()


# make predictions on test set
#predictions = model.predict(X_test)




# display images and predictions
#for i in range(10):
#    plt.subplot(2, 5, i+1)
#    plt.imshow(X_test[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
#    plt.title("Predicted: %d" % np.argmax(predictions[i]))
#    plt.axis('off')
#plt.show()