import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

folder = "images"

# .png .jpg .jpeg only for opencv
# doesnt join in alpha order, but is joined in a specific order
images = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith('.png')]

# print("Images:" ,images)


def get_avg_color(img):
    img = cv2.imread(img)
    img = img[50:-50, 50:-50]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img_rgb)
    # plt.title("fun pic")
    # plt.show()
    avg_color = img.mean(axis=(0,1))
    return avg_color

colors = {img: get_avg_color(img) for img in images}
for img, color in colors.items():
    print(os.path.basename(img), color)

def color_dist(c1, c2):
    return np.linalg.norm(c1-c2)

query_img = images[6]
query_color = colors[query_img]

distances = {img: color_dist(query_color, col) for img, col in colors.items() if img != query_img}

sim = sorted(distances.items(), key= lambda x: x[1]) [:2]
print("most similar: ", sim)

plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(cv2.imread(query_img), cv2.COLOR_BGR2RGB))
plt.title("query")

for i, (path, _) in enumerate(sim):
    plt.subplot(1,3,i+2)
    plt.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    plt.title(f"rec {i+1}")

plt.tight_layout()
plt.show()