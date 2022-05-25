from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np
import os
from PIL import Image
import cv2

datagen = ImageDataGenerator(        
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))

image_directory = r'data/hgs_flush/'
SIZE = 224
dataset = []
my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):    
    if (image_name.split('.')[1] == 'bmp'):        
        image = io.imread(image_directory + image_name)  
        print(image.size)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      
        image = Image.fromarray(image)        
        image = image.resize((SIZE,SIZE)) 
        dataset.append(np.array(image))
x = np.array(dataset)
i = 0
for batch in datagen.flow(x, batch_size=16,
                          save_to_dir= r'data/augmented_flush',
                          save_prefix='dr',
                          save_format='bmp'):    
    i += 1    
    if i > 50:        
        break