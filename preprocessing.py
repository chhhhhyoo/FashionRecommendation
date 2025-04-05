import pandas as pd
import numpy as np
import cv2
import os

path = 'data/vali_modified.csv'
df = pd.read_csv(path)

image_path_array = df['image_path'].to_numpy()
label_array = df['category'].to_numpy()
x1 = df['x1'].to_numpy().astype(np.float32)
y1 = df['y1'].to_numpy().astype(np.float32)
x2 = df['x2'].to_numpy().astype(np.float32)
y2 = df['y2'].to_numpy().astype(np.float32)

for i in range(len(image_path_array)):
    path = os.path.join("..", "img", image_path_array[i])
    img = cv2.imread(path)
    if img is None:
        continue
    h, w = img.shape[:2]

    x1[i] = x1[i] / w
    x2[i] = x2[i] / w
    y1[i] = y1[i] / h
    y2[i] = y2[i] / h

df['x1_modified'] = pd.DataFrame(x1)
df['y1_modified'] = pd.DataFrame(y1)
df['x2_modified'] = pd.DataFrame(x2)
df['y2_modified'] = pd.DataFrame(y2)

df.to_csv('data/vali_modified2.csv', index=False)
print(df.head())
