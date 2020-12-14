from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ortools.sat.python import cp_model

from enum import Enum
from random import randint
maxDim = 10


class Room(Enum):
    DININGROOM = 1
    KITCHEN = 2
    MINOR_BATHROOM = 3
    OTHER = 4


class BuildingSide(Enum):
    LANDSCAPE = 1
    OPEN = 2
    NONE = 3


rightSide = BuildingSide.LANDSCAPE
leftSide = BuildingSide.OPEN
topSide = BuildingSide.NONE
bottomSide = BuildingSide.NONE


class Rectangle:
    roomId = 1

    def __init__(self, roomType, minArea=1, width=0, height=0, adjacentTo=-1):
        # Name the variable names in the model properly.
        self.width = model.NewIntVar(
            1, maxDim, 'Width, room: %d' % Rectangle.roomId)
        self.height = model.NewIntVar(
            1, maxDim, 'Height, room: %d' % Rectangle.roomId)
        self.area = model.NewIntVar(
            minArea, maxDim * maxDim, 'Area, room: %d' % Rectangle.roomId)
        self.startRow = model.NewIntVar(
            0, maxDim, 'Starting row, room: %d' % Rectangle.roomId)
        self.startCol = model.NewIntVar(
            0, maxDim, 'Starting col, room: %d' % Rectangle.roomId)
        self.endRow = model.NewIntVar(
            0, maxDim, 'Ending row, room: %d' % Rectangle.roomId)
        self.endCol = model.NewIntVar(
            0, maxDim, 'Ending col, room: %d' % Rectangle.roomId)
        self.roomType = roomType

        self.addGenericConstraints(width, height)
        self.adjacentTo = adjacentTo
        Rectangle.roomId += 1

    def addGenericConstraints(self, width, height):
        # We call the methods if the method is invoked with these parameters.
        if(width > 0):
            self.addWidth(width)
        if(height > 0):
            self.addHeight(height)

        model.Add(self.width == self.endCol-self.startCol)
        model.Add(self.height == self.endRow-self.startRow)
        model.AddMultiplicationEquality(self.area, [self.width, self.height])

    def addRoomConstraints(self):
        adjacentTo = self.adjacentTo
        if self.roomType == Room.DININGROOM:
            for i in range(len(rooms)):
                if(rooms[i].roomType == Room.KITCHEN):
                    adjacentTo = i
        if adjacentTo != -1:
            AddAdjacencyConstraint(self, rooms[adjacentTo])

    def roomExistsWithinColumns(self, startCol, endCol):

        AddIntersectionBetweenEdges(
            [self.startCol, self.endCol], [startCol, endCol])

    def roomExistsWithinRows(self, startRow, endRow):
        AddIntersectionBetweenEdges(
            [self.startRow, self.endRow], [startRow, endRow])

    # def addDiningRoomConstraints(self):
    #     for room in rooms:
    #         if room.roomType == Room.KITCHEN:
    #             self.roomExistsWithinColumns(room.startCol, room.endCol)
    #             self.roomExistsWithinRows(room.startRow, room.endRow)
    #             break

    def toString(self):
        print("Rectangle coordinates: (%d,%d)" %
              (self.startRow, self.startCol))
        print("Rectangle width: %d, Rectangle height: %d" %
              (self.width, self.height))

    def addWidth(self, width):
        model.Add(self.width == width)

    def addHeight(self, height):
        model.Add(self.height == height)

    def getLeft(self):
        return self.startCol

    def getRight(self):
        return self.endCol

    def getTop(self):
        return self.startRow

    def getBottom(self):
        return self.endRow


def AddIntersectionBetweenEdges(a, b):
    l1 = a[0]
    r1 = a[1]
    l2 = b[0]
    r2 = b[1]
    l = model.NewIntVar(0, maxDim, '')  # l>=l1 and l>=l2
    model.AddMaxEquality(l, [l1, l2])
    r = model.NewIntVar(0, maxDim, '')  # r<=r1 and r<=r2
    model.AddMinEquality(r, [r1, r2])
    model.Add(l <= r)


def VisualizeApartments(apartment, rooms):
    visualizedApartment = [[0 for i in range(solver.Value(
        apartment.width)+10)] for j in range(solver.Value(apartment.height))]
    print(solver.Value(apartment.width), solver.Value(apartment.height))
    for index, room in enumerate(rooms):
        startRow = solver.Value(room.startRow)
        startCol = solver.Value(room.startCol)
        roomHeight = solver.Value(room.height)
        roomWidth = solver.Value(room.width)
        # if room.roomType == Room.DININGROOM or room.roomType == Room.KITCHEN:
        print(index + 1, room.roomType, startRow,
              solver.Value(room.endRow), startCol, solver.Value(room.endCol))
        for i in range(startRow, startRow + roomHeight):
            for j in range(startCol, startCol + roomWidth):
                visualizedApartment[i][j] = index + 1

    for row in visualizedApartment:
        print(row)

# This method sets the relation between the start and end (rows/columns)
# by adding the |AddNoOverlap2D| constraint to the model.


def AddNoIntersectionConstraint(rooms):
    rowIntervals = [model.NewIntervalVar(
        room.getTop(), room.height, room.getBottom(), 'room %d' % (roomNum + 1)) for roomNum, room in enumerate(rooms)]
    colIntervals = [model.NewIntervalVar(
        room.getLeft(), room.width, room.getRight(), 'room %d' % (roomNum + 1)) for roomNum, room in enumerate(rooms)]
    # how could this be optimized?
    model.AddNoOverlap2D(colIntervals, rowIntervals)


def GetBorders(rooms):
    leftBorders = [rooms[i].getLeft() for i in range(nOfRooms)]
    rightBorders = [rooms[i].getRight() for i in range(nOfRooms)]
    topBorders = [rooms[i].getTop() for i in range(nOfRooms)]
    bottomBorders = [rooms[i].getBottom() for i in range(nOfRooms)]
    return leftBorders, rightBorders, topBorders, bottomBorders


def ConstraintApartmentDimensions(apartment):
    leftBorders, rightBorders, topBorders, bottomBorders = GetBorders(rooms)

    model.AddMinEquality(apartment.getLeft(), leftBorders)
    model.AddMaxEquality(apartment.getRight(), rightBorders)
    model.AddMinEquality(apartment.getTop(), topBorders)
    model.AddMaxEquality(apartment.getBottom(), bottomBorders)


def AddAdjacencyConstraint(room, adjacentRoom):
    room.roomExistsWithinColumns(adjacentRoom.startCol, adjacentRoom.endCol)
    room.roomExistsWithinRows(adjacentRoom.startRow, adjacentRoom.endRow)

########################   Main Method Starts Here   ########################

########################   Process Future Input Here ########################


nOfApartments = 1
nOfRooms = 5
rooms = []


model = cp_model.CpModel()
minArea = [randint(1, 5) for i in range(nOfRooms)]
# minArea = [4, 4, 1, 4, 4]
print(minArea)
for i in range(nOfRooms):
    roomType = Room.OTHER
    if i == 1:
        roomType = Room.DININGROOM
    elif i == 0:
        roomType = Room.KITCHEN
    elif i == 2:
        roomType = Room.MINOR_BATHROOM

    rooms.append(
        Rectangle(roomType, minArea[i], adjacentTo=(0 if i == 2 else -1)))

########################   Process Future Input Here ########################
for room in rooms:
    room.addRoomConstraints()

AddNoIntersectionConstraint(rooms)

apartment = Rectangle(Room.OTHER)

ConstraintApartmentDimensions(apartment)

model.Minimize(apartment.area)
solver = cp_model.CpSolver()
status = solver.Solve(model)
print(solver.StatusName())
print(solver.Value(apartment.area))
print('time = ', solver.WallTime())
VisualizeApartments(apartment, rooms)

########################   Main Method Ends Here   ##########################
