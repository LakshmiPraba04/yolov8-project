#Preprocessing Script Inscription Image
# Skew Correctio Function

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import json
import re
from sklearn.metrics import accuracy_score
import seaborn as sns

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = cv2.rotate(arr, angle)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated

image = cv2.imread('3.png')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

# Skew correction
angle, rotated = correct_skew(image)
print(angle)
cv2.imwrite('rotated.jpg', rotated)


# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()

# Median blur
filter1 = cv2.medianBlur(gray, 5)

# Gaussian blur
filter2 = cv2.GaussianBlur(filter1, (5, 5), 0)

# Denoising
dst = cv2.fastNlMeansDenoising(filter2, None, 17, 9, 17)
# plt.imshow(dst, cmap='gray')
plt.show()

# Binarization
th1 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
plt.imshow(th1, cmap='gray')
plt.show()

# Save preprocessed image
cv2.imwrite('PreProcessed.jpg',th1)