import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

folder = "images"
images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print("Images:", images)

# img = cv2.imread(images[0])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img_rgb)
# plt.title("fun pic")
# plt.show()

def get_avg_color(img_path):
    img = cv2.imread(img_path)
    img = img[50:-50, 50:-50]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    avg_color = img.mean(axis=(0, 1))
    return avg_color

colors = {img: get_avg_color(img) for img in images}

for img_path, color in colors.items():
    print(os.path.basename(img_path), color)

def color_distance(c1, c2 ):
    return np.linalg.norm(c1-c2)

query_img = images[0]
query_color = colors[query_img]

distances = {img: color_distance(query_color, c) for img, c in colors.items() if img != query_img}

similar = sorted(distances.items(), key= lambda x: x[1]) [:2]
print("Most similar:", similar)

plt.figure(figsize = (10, 3))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(cv2.imread(query_img), cv2.COLOR_BGR2RGB))
plt.title("Query")

for i, (path, _) in enumerate(similar):
    plt.subplot(1, 3, i + 2)
    plt.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    plt.title(f"Recommendation {i + 1}")

plt.tight_layout()
plt.show()
