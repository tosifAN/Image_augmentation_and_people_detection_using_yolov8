import numpy as np
import imageio
import imgaug as ia
from imgaug import augmenters as iaa

# Read the image
image = imageio.imread('input/sample.jpg')

# Define the augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Affine(rotate=(-45, 45)),  # random rotations
    iaa.GaussianBlur(sigma=(0.0, 3.0)),  # random Gaussian blur
    iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),  # random Gaussian noise
])

# Number of augmented images to generate
num_augmented_images = 10

# Generate and save multiple augmented images
for i in range(num_augmented_images):
    augmented_image = seq(image=image)
    # Save each augmented image with a unique name
    imageio.imwrite(f'preview/augmented_image_{i}.jpg', augmented_image)
