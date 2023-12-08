import tensorflow as tf
import os
import numpy as np

# Read the image
image_path = 'input/sample.jpg'
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image, channels=3)  # Decode the image (change channels as needed)
IMG_SIZE = 180
# Define augmentation functions using tf.image
def augment_image(image):
    seed = (2, 0) 
    # Randomly flip the image horizontally
    image = tf.image.random_flip_left_right(image)
    #image=tf.image.stateless_random_flip_up_down(image) error


  #  image = tf.image.stateless_random_crop(
   #   image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed) buri image
    
    # Randomly adjust brightness and contrast
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    
    # Randomly rotate the image
    image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    return image

# Create a folder to save augmented images
output_folder = 'preview/tensor'
os.makedirs(output_folder, exist_ok=True)

# Number of augmented images to generate
num_augmented_images = 10

# Generate and save augmented images
for i in range(num_augmented_images):
    augmented_image = augment_image(image)
    
    # Save augmented images to the output folder with unique names
    image_filename = f'augmented_image_{i}.jpg'
    image_path = os.path.join(output_folder, image_filename)
    
    # Convert the TensorFlow tensor to a NumPy array and save as an image
    image_np = tf.keras.preprocessing.image.img_to_array(augmented_image)
    image_np = image_np.astype(np.uint8)
    tf.keras.preprocessing.image.save_img(image_path, image_np)

print(f'{num_augmented_images} augmented images saved to {output_folder}.')
