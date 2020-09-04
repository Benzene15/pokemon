from cv2 import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS=1000
IMG_WIDTH=30
IMG_HEIGHT=30
NUM_CATEGORIES=150
TEST_SIZE=0.4

def main():
    images,labels,num=load_data("PokemonData")
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    print("n: "+ str(num))
    if(len(sys.argv)==2):
        filename = sys.argv[1]
        model.save(filename)
        print("Model saved to {filename}.")


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
    model= tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(32, (5,5), input_shape=(IMG_WIDTH,IMG_WIDTH,3)),
        tf.keras.layers.MaxPooling2D(pool_size=(5,5)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),



        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
        
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()