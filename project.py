from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ortools.sat.python import cp_model


class Rectangle:
    def __init__(self, minArea, model, width=0, height=0):
        self.minArea = minArea
        self.width = model.NewIntVar(1, 1000, 'w')
        self.height = model.NewIntVar(1, 1000, 'h')
        self.area = model.NewIntVar(minArea, 1000*1000, 'area')
        self.startRow = model.NewIntVar(0, 1000, 'startRow')
        self.startCol = model.NewIntVar(0, 1000, 'startCol')
        if(width > 0):
            self.addWidth(width, model)
        if(height > 0):
            self.addHeight(height, model)
        model.AddMultiplicationEquality(self.area, [self.width, self.height])

        model.Add(self.area >= minArea)
    # def __init__(self, startX, startY, width, height):
    #     self.width = width
    #     self.height = height

    def toString(self):
        print("Rectangle coordinates: (%d,%d)" % (self.startX, self.startY))
        print("Rectangle width: %d, Rectangle height: %d" %
              (self.width, self.height))

    def addWidth(self, width, model):
        model.Add(self.width == width)

    def addHeight(self, height, model):
        model.Add(self.height == height)

    def getLeft(self):
        return self.startCol

    def getRight(self):
        return self.getLeft()+self.width-1

    def getTop(self):
        return self.startRow

    def getBottom(self):
        return self.getTop()+self.height-1


def addNoIntersectionConstraint(model, rooms):
    for i in range(len(rooms)):
        for j in range(i):
            A = rooms[i]
            B = rooms[j]
            model.Add(A.getLeft() < B.getRight()
                      and A.getRight() > B.getLeft() and A.getTop() > B.getBottom() and A.getBottom() < B.getTop())


nOfApartments = 1
nOfRooms = 4
rooms = []


# optional width, height => width, height
model = cp_model.CpModel()
minArea = [i for i in range(nOfRooms)]
for i in range(nOfRooms):
    rooms.append(Rectangle(minArea[i], model, 2 if i == 0 else 0))
# for i in start:
#     print("%s = %i" % (i, solver.Value(i)))
addNoIntersectionConstraint(model, rooms)

solver = cp_model.CpSolver()
status = solver.Solve(model)
print(solver.StatusName())
leftBorders = [rooms[i].getLeft() for i in range(nOfRooms)]
rightBorders = [rooms[i].getRight() for i in range(nOfRooms)]
upBorders = [rooms[i].getTop() for i in range(nOfRooms)]
downBorders = [rooms[i].getBottom() for i in range(nOfRooms)]
apartment_startRow = model.NewIntVar(0, 1000, 'appar_start_row')
apartment_endRow = model.NewIntVar(0, 1000, 'appar_end_row')
apartment_startCol = model.NewIntVar(0, 1000, 'appar_start_col')
apartment_endCol = model.NewIntVar(0, 1000, 'appar_end_scol')
model.AddMinEquality(apartment_startCol, leftBorders)
# model.AddMaxEquality(apartment_endCol, rightBorders)
model.AddMinEquality(apartment_startRow, upBorders)
# model.AddMinEquality(apartment_endRow, downBorders)
for room in rooms:
    print(solver.Value(room.startRow), solver.Value(room.startCol),
          solver.Value(room.width), solver.Value(room.height))
