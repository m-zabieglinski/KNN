import kohonen as KN
from math import sin, cos, pi

#creating a dataset of points representing a disk
disk = [[0, 0]]

size = [40, 10]

t = 0
for i in range(size[0]):
    r = 0.1
    for i in range(size[1]):
        disk.append([r * cos(-t), r * sin(-t)])
        r += 0.1
    t -= 2 * pi / size[0]

step = 0.9
epochs = 100

kn = KN.KN(size = 5)
kn.create()
kn.train(disk, epochs = epochs, step = step)
kn.showmap(disk)
# mapa = kn.categorize(disk)

        
