from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ortools.sat.python import cp_model

maxDim = 10


class Rectangle:
    def __init__(self, minArea, model, width=0, height=0):
        self.minArea = minArea
        self.width = model.NewIntVar(1, maxDim, 'w')
        self.height = model.NewIntVar(1, maxDim, 'h')
        self.area = model.NewIntVar(minArea, maxDim*maxDim, 'area')
        self.startRow = model.NewIntVar(0, maxDim, 'startRow')
        self.startCol = model.NewIntVar(0, maxDim, 'startCol')
        self.endRow = model.NewIntVar(0, maxDim, 'endRow')
        self.endCol = model.NewIntVar(0, maxDim, 'endCol')
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
        return self.endCol

    def getTop(self):
        return self.startRow

    def getBottom(self):
        return self.endRow


def VisualizeApartments(model, rooms):
    visualizedApartment = [[0 for i in range(10)] for j in range(10)]
    for index, room in enumerate(rooms):
        startRow = solver.Value(room.startRow)
        startCol = solver.Value(room.startCol)
        roomHeight = solver.Value(room.height)
        roomWidth = solver.Value(room.width)
        for i in range(startRow, startRow + roomHeight):
            for j in range(startCol, startCol + roomWidth):
                visualizedApartment[i][j] = index + 1

    for row in visualizedApartment:
        print(row)


def addNoIntersectionConstraint(model, rooms):
    # for i in range(len(rooms)):
    #     for j in range(i):
    #         A = rooms[i]
    #         B = rooms[j]
    #         model.Add(A.getLeft() < B.getRight()
    #                   and A.getRight() > B.getLeft() and A.getTop() > B.getBottom() and A.getBottom() < B.getTop())
    rowIntervals = [model.NewIntervalVar(
        room.getTop(), room.height, room.getBottom(), 'room 1 ') for room in rooms]
    colIntervals = [model.NewIntervalVar(
        room.getLeft(), room.width, room.getRight(), 'room 1 ') for room in rooms]
    model.AddNoOverlap2D(colIntervals, rowIntervals)


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


leftBorders = [rooms[i].getLeft() for i in range(nOfRooms)]
rightBorders = [rooms[i].getRight() for i in range(nOfRooms)]
upBorders = [rooms[i].getTop() for i in range(nOfRooms)]
downBorders = [rooms[i].getBottom() for i in range(nOfRooms)]
apartment_startRow = model.NewIntVar(0, maxDim, 'appar_start_row')
apartment_endRow = model.NewIntVar(0, maxDim, 'appar_end_row')
apartment_startCol = model.NewIntVar(0, maxDim, 'appar_start_col')
apartment_endCol = model.NewIntVar(0, maxDim, 'appar_end_scol')
model.AddMinEquality(apartment_startCol, leftBorders)
model.AddMaxEquality(apartment_endCol, rightBorders)
model.AddMinEquality(apartment_startRow, upBorders)
model.AddMaxEquality(apartment_endRow, downBorders)
apartment_width = model.NewIntVar(0, maxDim, 'xx')
model.Add(apartment_width == apartment_endCol-apartment_startCol)

apartment_height = model.NewIntVar(0, maxDim, 'xx')
model.Add(apartment_height == apartment_endRow-apartment_startRow)

appartment_area = model.NewIntVar(0, maxDim, 'area')
model.AddMultiplicationEquality(appartment_area, [
    apartment_width, apartment_height])

model.Minimize(appartment_area)
solver = cp_model.CpSolver()
status = solver.Solve(model)
print(solver.StatusName())
print(solver.Value(appartment_area))
for room in rooms:
    print(solver.Value(room.startRow), solver.Value(room.startCol),
          solver.Value(room.width), solver.Value(room.height))
print(solver.Value(appartment_area))
VisualizeApartments(model, rooms)
#
