import solt
import solt.transforms as slt
import imageio
import random
import tensorflow as tf
import os
import numpy as np




# Create a folder to save augmented images
output_folder = 'image_augmented'
os.makedirs(output_folder, exist_ok=True)

for j in range(1,31):
  print(j)
  input_img = imageio.imread(f'image/000000{j}.jpg')

  h, w, c = input_img.shape
  img = input_img[:w]




  stream = solt.Stream([
    slt.Rotate(angle_range=(-90, 90), p=1, padding='r'),
    slt.Flip(axis=1, p=0.5),
    slt.Flip(axis=0, p=0.5),
    slt.Shear(range_x=0.3, range_y=0.8, p=0.5, padding='r'),
    slt.Scale(range_x=(0.8, 1.3), padding='r', range_y=(0.8, 1.3), same=False, p=0.5),
    slt.Pad((w, h), 'r'),
    slt.Crop((w, w), 'r'),
    slt.Blur(k_size=7, blur_type='m'),
    solt.SelectiveStream([
        slt.CutOut(40, p=1),
        slt.CutOut(50, p=1),
        slt.CutOut(10, p=1),
        solt.Stream(),
        solt.Stream(),
      ], n=3),
  ], ignore_fast_mode=True)

  n_augs = 5
  random.seed(2)
  for i in range(n_augs):
    img_aug = stream({'image': img}, return_torch=False, ).data[0].squeeze()

    # Save augmented images to the output folder with unique names
    image_filename = f'augmented_image_{j}{i}.jpg'
    image_path = os.path.join(output_folder, image_filename)
    
    #Convert the TensorFlow tensor to a NumPy array and save as an image
    image_np = tf.keras.preprocessing.image.img_to_array(img_aug)
    image_np = image_np.astype(np.uint8)
    tf.keras.preprocessing.image.save_img(image_path, image_np)
