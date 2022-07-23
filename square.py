import kohonen as KN

#creating a dataset of points representing a square
square = []

size = [10, 10]

x = 0
for i in range(size[0]):
    x += 1
    y = 0
    for i in range(size[1]):
        y += 1
        square.append([x, y])
 
step = 0.9
epochs = 100

kn = KN.KN(size = 3)
kn.create()
kn.train(square, epochs = epochs, step = step)
kn.showmap(square)
# mapa = kn.categorize(disk)

        
