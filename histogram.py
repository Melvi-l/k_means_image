import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import math

# Charger l'image
image_path = "parrot.jpg"
image = Image.open(image_path)

# resize_factor = 200 / max(image.width, image.height)
# image = image.resize((int(image.size[0]*resize_factor), int(image.size[1]*resize_factor)))
# image = image.resize((350,350))
image = image.crop((0, 0, 350, 350))
image = image.resize((100,100))
plt.imshow(image)
plt.show()

# Obtenir les données des pixels sous forme de tableau numpy
pixels = np.array(image)

# Récupérer les dimensions de l'image
height, width, channels = pixels.shape

# Créer des tableaux pour les coordonnées x, y, z et les couleurs
x = pixels[:, :, 0].flatten()  # canal rouge
y = pixels[:, :, 1].flatten()  # canal vert
z = pixels[:, :, 2].flatten()  # canal bleu


colors = pixels.reshape((height * width, channels)) / 255.0  # Normaliser les valeurs des couleurs à l'intervalle [0, 1]

# Créer la figure 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Afficher le nuage de points
ax.scatter(x, y, z, c=colors, marker='.', s=1)

# Définir les étiquettes des axes
ax.set_xlabel('Rouge')
ax.set_ylabel('Vert')
ax.set_zlabel('Bleu')


# Afficher le graphique
plt.show()