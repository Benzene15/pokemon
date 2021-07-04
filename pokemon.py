from cv2 import cv2
import numpy as np
import os
import os.path  
from os import path
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout

from sklearn.model_selection import train_test_split

EPOCHS=50
IMG_WIDTH=30
IMG_HEIGHT=30
NUM_CATEGORIES=150
TEST_SIZE=0.3

def main():

    #if(path.exists("pokemon.h5")):

    #Load the image data
    images,labels,num=load_data("PokemonData")

    #Split the data
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    #Make the model
    model = get_model()

    #Train the model
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32)

    #Evaluate the model
    model.evaluate(x_test,  y_test, verbose=2)

    print("n: "+ str(num))

    #If there is a second arg in commond line save to it
    #end file with .h5
    if(len(sys.argv)==2):
        filename = sys.argv[1]
        model.save(filename)
        print("Model saved to c"+filename+".")


def load_data(data_dir):
    imgs=[]
    labels=[]

    i=0
    n=0
    for folder in os.listdir(data_dir):
        f=os.path.join(data_dir,folder)
        for file in os.listdir(f):
            img=cv2.imread(os.path.join(f,file))
            if(img is not None):
                img=cv2.resize(img,(IMG_HEIGHT,IMG_WIDTH))
                imgs.append(img)
                labels.append(i)
                print(file)
                n+=1
        i+=1        
    return imgs, labels,n

def get_model():
    model = Sequential([
                    Conv2D(filters=128,  kernel_size=(3,3), activation="sigmoid", input_shape=(IMG_WIDTH,IMG_WIDTH,3)),
                    MaxPool2D(2,2, padding='same'),
                    BatchNormalization(),
                    Dropout(0.4),
                    
                    Conv2D(filters=256, kernel_size=(3,3), activation = "sigmoid"),
                    MaxPool2D(2,2, padding="same"),
                    BatchNormalization(),
                    Dropout(0.4),
                    
                    Flatten(),

                    Dense(units = 4096, activation="relu", input_shape=(IMG_WIDTH,IMG_WIDTH,3)),
                    Dropout(0.2),

                    Dense(units = 2048, activation = "relu"),
                    Dropout(0.2),

                    Dense(units = 4096, activation="relu"),
                    Dropout(0.2),

                    Dense(units = 1024, activation = "relu"),
                    Dropout(0.2),

                    Dense(units = 512, activation = "relu"),
                    Dropout(0.2),
                    
                    Dense(NUM_CATEGORIES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(model.summary())

    return model


if __name__ == "__main__":
    main()