import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


im = Image.open('dataset/jaffe/KA.AN2.40.tiff')
(w, h) = im.size
bags = []

pos_patch = lambda n: [
    (0, 0, n//2, n//2), (n//2, 0, n, n//2),
    (0, n//2, n//2, n), (n//2, n//2, n, n)
]

sub_image = lambda image, dim: [image.crop(patch) for patch in pos_patch(dim)]

bags += sub_image(im, w)
depth_bags = []
temp_bags = bags[:]

for i in range(2):
    for s_img in temp_bags:
        sub = sub_image(s_img, w//2**i)
        bags += sub
        depth_bags += sub
    
    temp_bags = depth_bags[:]
    depth_bags = []

def cal_mean(image):
    values = [i for i in image.getdata()]
    n = len(values)

    return sum(values)/n

means = lambda bags: [cal_mean(image) for image in bags]

fig, axs = plt.subplots(nrows=4**2, ncols=4**2, figsize=(8, 8),
                        subplot_kw={'xticks': [], 'yticks': []})

fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.3)

for ax, i in zip(axs.flat, range(len(bags))):
    ax.imshow(bags[i], cmap='gray')
    # ax.set_title(str(i))

# plt.plot(bags)
plt.tight_layout()
plt.show()