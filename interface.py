from PIL import Image
import numpy as np

im = Image.open('Carte.png')
im.show()
im_array = np.array(im)
