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
    OTHER = 3


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

    def __init__(self, roomType, minArea=1, width=0, height=0):
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

        self.addRoomConstraints()

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
        if self.roomType == Room.DININGROOM:
            self.addDiningRoomConstraints()

    def roomExistsWithinColumns(self, startCol, endCol):
        b1 = model.NewBoolVar('')
        b2 = model.NewBoolVar('')
        b3 = model.NewBoolVar('')
        b4 = model.NewBoolVar('')
        # model.Add(self.startCol <= endCol and self.endCol >=
        #           endCol).OnlyEnforceIf(b1)
        b11 = model.NewBoolVar('')
        model.Add(self.startCol <= endCol).OnlyEnforceIf(b11)
        b12 = model.NewBoolVar('')
        model.Add(self.endCol >=
                  endCol).OnlyEnforceIf(b12)
        model.AddBoolAnd([b11, b12]).OnlyEnforceIf(b1)
        b21 = model.NewBoolVar('')
        b22 = model.NewBoolVar('')
        model.Add(self.startCol >= startCol).OnlyEnforceIf(b21)
        model.Add(self.startCol <=
                  endCol).OnlyEnforceIf(b22)
        model.AddBoolAnd([b21, b22]).OnlyEnforceIf(b2)
        b31 = model.NewBoolVar('')
        b32 = model.NewBoolVar('')
        model.Add(self.endCol >= startCol).OnlyEnforceIf(b31)
        model.Add(self.endCol <=
                  endCol).OnlyEnforceIf(b32)
        model.AddBoolAnd([b31, b32]).OnlyEnforceIf(b3)
        b41 = model.NewBoolVar('')
        b42 = model.NewBoolVar('')
        model.Add(startCol >= self.startCol).OnlyEnforceIf(b41)
        model.Add(startCol <=
                  self.endCol).OnlyEnforceIf(b42)
        model.AddBoolAnd([b41, b42]).OnlyEnforceIf(b4)
        model.AddBoolOr([b1, b2, b3, b4])

    def roomExistsWithinRows(self, startRow, endRow):
        b1 = model.NewBoolVar('')
        b2 = model.NewBoolVar('')
        b3 = model.NewBoolVar('')
        b4 = model.NewBoolVar('')
        b11 = model.NewBoolVar('')
        b12 = model.NewBoolVar('')
        model.Add(self.startRow <= endRow).OnlyEnforceIf(b11)
        model.Add(self.endRow >=
                  endRow).OnlyEnforceIf(b12)
        model.AddBoolAnd([b11, b12]).OnlyEnforceIf(b1)
        b21 = model.NewBoolVar('')
        b22 = model.NewBoolVar('')
        model.Add(self.startRow >= startRow).OnlyEnforceIf(b21)
        model.Add(self.startRow <=
                  endRow).OnlyEnforceIf(b22)
        model.AddBoolAnd([b21, b22]).OnlyEnforceIf(b2)
        b31 = model.NewBoolVar('')
        b32 = model.NewBoolVar('')
        model.Add(self.endRow >= startRow).OnlyEnforceIf(b31)
        model.Add(self.endRow <=
                  endRow).OnlyEnforceIf(b32)
        model.AddBoolAnd([b31, b32]).OnlyEnforceIf(b3)
        b41 = model.NewBoolVar('')
        b42 = model.NewBoolVar('')
        model.Add(startRow >= self.startRow).OnlyEnforceIf(b41)
        model.Add(startRow <=
                  self.endRow).OnlyEnforceIf(b42)
        model.AddBoolAnd([b41, b42]).OnlyEnforceIf(b4)
        model.AddBoolOr([b1, b2, b3, b4])

    def addDiningRoomConstraints(self):
        for room in rooms:
            if room.roomType == Room.KITCHEN:
                self.roomExistsWithinColumns(room.startCol, room.endCol)
                self.roomExistsWithinRows(room.startRow, room.endRow)
                break

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

########################   Main Method Starts Here   ########################

########################   Process Future Input Here ########################


nOfApartments = 1
nOfRooms = 5
rooms = []


model = cp_model.CpModel()
minArea = [randint(1, 5) for i in range(nOfRooms)]
print(minArea)
for i in range(nOfRooms):
    roomType = Room.OTHER
    if i == 1:
        roomType = Room.DININGROOM
    elif i == 0:
        roomType = Room.KITCHEN
    rooms.append(
        Rectangle(roomType, minArea[i]))


########################   Process Future Input Here ########################

AddNoIntersectionConstraint(rooms)

apartment = Rectangle(Room.OTHER)

ConstraintApartmentDimensions(apartment)

model.Minimize(apartment.area)
solver = cp_model.CpSolver()
status = solver.Solve(model)
print(solver.StatusName())
print(solver.Value(apartment.area))

VisualizeApartments(apartment, rooms)

########################   Main Method Ends Here   ##########################
