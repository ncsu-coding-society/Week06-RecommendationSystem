import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# heiley heuli

folder = "images"

# creates a list of all image file paths in the specified folder (Note: in order)
images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print("Images: ", images)

def get_avg_color(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    avg_color = img_rgb.mean(axis=(0, 1)) # axis 0 = height, 1 = width, this averages rbg value over all pixels
    return avg_color
    
def color_distance(c1, c2):
    return np.linalg.norm(c1-c2)

colors = { image: get_avg_color(image) for image in images }

for img_path, rgb_array in colors.items():
    print(os.path.basename(img_path), rgb_array)

query_img = images[4]
query_color = colors[query_img]

distances = {img: color_distance(query_color, c) for img, c in colors.items() if img != query_img }

similar = sorted(distances.items(), key= lambda x: x[1]) [:4]
print("Most similar: ", similar)

plt.figure(figsize = (10, 3))

plt.subplot(1, 5, 1)

plt.imshow(cv2.cvtColor(cv2.imread(query_img), cv2.COLOR_BGR2RGB))
plt.title("Query")

for i, (path, _) in enumerate(similar):
    plt.subplot(1, 5, i + 2)
    plt.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    plt.title(f"Recommendation {i + 1}")
    
plt.tight_layout()
plt.show()