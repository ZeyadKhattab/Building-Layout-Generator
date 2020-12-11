from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ortools.sat.python import cp_model

from enum import Enum

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
        if self.startCol <= endCol and self.endCol >= endCol:
            return True
        if self.startCol >= startCol and self.startCol <= endCol:
            return True
        if self.endCol >= startCol and self.endCol <= endCol:
            return True
        if startCol >= self.startCol and startCol <= self.endCol:
            return True
        return False
    
    def roomExistsWithinRows(self, startRow, endRow):
        if self.startRow <= endRow and self.endRow >= endRow:
            return True
        if self.startRow >= startRow and self.startRow <= endRow:
            return True
        if self.endRow >= startRow and self.endRow <= endRow:
            return True
        if startRow >= self.startRow and startRow <= self.endRow:
            return True
        return False

    def addDiningRoomConstraints(self):
        for room in rooms:
            if room.roomType == Room.KITCHEN:
                #print(self.roomType, room.roomType)
                # Once this constraints starts to work, add conditions for columns just like the row ones. 
                #model.Add((self.startRow == room.endRow and self.roomExistsWithinColumns(room.startCol, room.endCol)) or (
                #    self.endRow == room.startRow and self.roomExistsWithinColumns(room.startCol, room.endCol)))
                #flagRows = model.NewBoolVar('rows')
                #flagColumns = model.NewBoolVar('columns')
                #model.Add(flagRows == 1).OnlyEnforceIf(self.roomExistsWithinRows(room.startRow, room.endRow) == True)
                #model.Add(flagColumns == 1).OnlyEnforceIf(self.roomExistsWithinColumns(room.startCol, room.endCol) == True)
                model.Add(self.startCol == room.endCol or self.endCol == room.startCol).OnlyEnforceIf(self.roomExistsWithinRows(room.startRow, room.endRow) == True)
                model.Add(self.startRow == room.endRow or self.endRow == room.startRow).OnlyEnforceIf(self.roomExistsWithinColumns(room.startCol, room.endCol) == True)
                #model.Add(flagRows + flagColumns >= 1)
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
        #if room.roomType == Room.DININGROOM or room.roomType == Room.KITCHEN:
        print(index + 1, room.roomType, startRow, solver.Value(room.endRow), startCol, solver.Value(room.endCol))
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
minArea = [i for i in range(nOfRooms)]

for i in range(nOfRooms):
    roomType = Room.OTHER
    if i == 1:
        roomType = Room.DININGROOM
    elif i == 0:
        roomType = Room.KITCHEN
    rooms.append(
        Rectangle(roomType, minArea[i], 2 if i == 0 else 0, 3 if i == 1 else 2))


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
