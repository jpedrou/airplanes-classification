import cv2
import imghdr
import os
import matplotlib.pyplot as plt

# Package directory to the images
data_dir = "../../data/raw"

# Acceptable extensions
image_extensions = ["jpg", "jpeg", "bmp", "png"]

# =========================================================
# Remove strange images
# =========================================================

# Plot image
img = cv2.imread(os.path.join(data_dir, 'commercial-airplaines', 'aircraft-a350-neo-closeup-sas-5.jpg'))
img.shape
type(img)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_extensions:
                print('Image not in extensions list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)