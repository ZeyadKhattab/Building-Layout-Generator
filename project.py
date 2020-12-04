class Rectangle:
    def __init__(self, minArea):
        self.minArea = minArea

    def __init__(self, startX, startY, width, height):
        self.startX = startX
        self.startY = startY
        self.width = width
        self.height = height

    def toString(self):
        print("Rectangle coordinates: (%d,%d)" %(self.startX, self.startY))
        print("Rectangle width: %d, Rectangle height: %d" %(self.width, self.height))


nOfApartments = 1
nOfRooms = 4
rooms = []

# optional width, height => width, height
#for i in range(0, nOfRooms):


rectangle = Rectangle(3,2,1,4)
rectangle.X = 2
print(rectangle.X)
rectangle.toString()