from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ortools.sat.python import cp_model

import matplotlib.pyplot as plt

from enum import Enum
from random import randint
maxDim = 10
DI = [-1, 1, 0, 0]
DJ = [0, 0, -1, 1]


class Room(Enum):
    DININGROOM = 1
    KITCHEN = 2
    MINOR_BATHROOM = 3
    DRESSING_ROOM = 4
    BEDROOM = 5
    SUNROOM = 6
    CORRIDOR = 7
    OTHER = 8


class BuildingSide(Enum):
    LANDSCAPE = 1
    OPEN = 2
    NONE = 3


ROOM_TYPE_MAP = {'Room.DININGROOM': 'DR', 'Room.KITCHEN': 'KT', 'Room.MINOR_BATHROOM': 'MB',
               'Room.DRESSING_ROOM': 'DRS', 'Room.BEDROOM': 'BD', 'Room.SUNROOM': 'SR', 'Room.CORRIDOR': 'C', 'Room.OTHER': 'X'}


FLOOR_RIGHT_SIDE = BuildingSide.OPEN
FLOOR_LEFT_SIDE = BuildingSide.LANDSCAPE
FLOOR_TOP_SIDE = BuildingSide.LANDSCAPE
FLOOR_BOTTOM_SIDE = BuildingSide.NONE


class Rectangle:
    roomId = 1

    def __init__(self, room_type, min_area=1, width=0, height=0, adjacent_to=-1, apartment=1):
        # Name the variable names in the model properly.
        self.width = model.NewIntVar(
            1, maxDim, 'Width, room: %d' % Rectangle.roomId)
        self.height = model.NewIntVar(
            1, maxDim, 'Height, room: %d' % Rectangle.roomId)
        self.area = model.NewIntVar(
            min_area, maxDim * maxDim, 'Area, room: %d' % Rectangle.roomId)
        self.startRow = model.NewIntVar(
            0, maxDim, 'Starting row, room: %d' % Rectangle.roomId)
        self.startCol = model.NewIntVar(
            0, maxDim, 'Starting col, room: %d' % Rectangle.roomId)
        self.endRow = model.NewIntVar(
            0, maxDim, 'Ending row, room: %d' % Rectangle.roomId)
        self.endCol = model.NewIntVar(
            0, maxDim, 'Ending col, room: %d' % Rectangle.roomId)
        self.room_type = room_type
        self.apartment = apartment

        self.addGenericConstraints(width, height)
        self.adjacent_to = adjacent_to
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

    def addRoomConstraints(self, apartment):
        adjacent_to = self.adjacent_to
        if self.room_type == Room.DININGROOM:
            for i in range(len(apartment)):
                if(apartment[i].room_type == Room.KITCHEN):
                    adjacent_to = i
        if adjacent_to != -1:
            AddAdjacencyConstraint(self, apartment[adjacent_to])

    def roomExistsWithinColumns(self, startCol, endCol):
        return AddIntersectionBetweenEdges(
            [self.startCol, self.endCol], [startCol, endCol])

    def roomExistsWithinRows(self, startRow, endRow):
        return AddIntersectionBetweenEdges(
            [self.startRow, self.endRow], [startRow, endRow])

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
    eq = model.NewBoolVar('')
    l = model.NewIntVar(0, maxDim, '')
    model.AddMaxEquality(l, [l1, l2])
    r = model.NewIntVar(0, maxDim, '')
    model.AddMinEquality(r, [r1, r2])
    leq = model.NewBoolVar('')
    model.Add(l <= r).OnlyEnforceIf(leq)
    model.Add(l > r).OnlyEnforceIf(leq.Not())
    model.Add(l == r).OnlyEnforceIf(eq)
    model.Add(l != r).OnlyEnforceIf(eq.Not())
    return leq, eq

# This method sets the relation between the start and end (rows/columns)
# by adding the |AddNoOverlap2D| constraint to the model.

# Takes in the flattened version of the apartments, universal.


def AddNoIntersectionConstraint(flattened_rooms):
    rowIntervals = [model.NewIntervalVar(
        room.getTop(), room.height, room.getBottom(), 'room %d' % (roomNum + 1)) for roomNum, room in enumerate(flattened_rooms)]
    colIntervals = [model.NewIntervalVar(
        room.getLeft(), room.width, room.getRight(), 'room %d' % (roomNum + 1)) for roomNum, room in enumerate(flattened_rooms)]
    # how could this be optimized?
    model.AddNoOverlap2D(colIntervals, rowIntervals)

def AddAdjacencyConstraint(room, adjacentRoom, add=1):
    columnsLeq, columnsEq = room.roomExistsWithinColumns(
        adjacentRoom.startCol, adjacentRoom.endCol)
    rowsLeq, rowsEq = room.roomExistsWithinRows(
        adjacentRoom.startRow, adjacentRoom.endRow)
    intersection = model.NewBoolVar('')
    model.Add(intersection == 1).OnlyEnforceIf([columnsLeq, rowsLeq])
    model.AddImplication(columnsLeq.Not(), intersection.Not())
    model.AddImplication(rowsLeq.Not(), intersection.Not())
    if add == 1:
        model.Add(intersection == 1)
    model.Add(columnsEq + rowsEq < 2)
    return intersection

# Takes in one single apartment, non-universal.


def AddCorridorConstraint(nOfCorridors, flattened_rooms):
    '''The last nOfCorriodors should have type corridor'''
    assert(nOfCorridors > 0)
    n = len(flattened_rooms)
    # All the corriods are adjacent to each other
    for i in range(n-nOfCorridors, n-1):
        AddAdjacencyConstraint(flattened_rooms[i], flattened_rooms[i+1])
    for i in range(n-nOfCorridors):
        currRoom = flattened_rooms[i]
        adjacent_to_corridors = []
        for j in range(n-nOfCorridors, n):
            corridor = flattened_rooms[j]
            adjacent_to_corridors.append(AddAdjacencyConstraint(
                currRoom, corridor, 0))
        model.Add(sum(adjacent_to_corridors) > 0)

# Takes in the flattened version of the apartments, universal.


def GetGrid(flattened_rooms):
    n = len(flattened_rooms)
    grid = [[model.NewIntVar(-1, n-1, '') for j in range(maxDim)]
            for i in range(maxDim)]
    for i in range(maxDim):
        for j in range(maxDim):
            intersections = []
            for index, room in enumerate(flattened_rooms):

                # rows
                greater_than_r1 = model.NewBoolVar('')
                less_than_r2 = model.NewBoolVar('')
                model.Add(i >= room.startRow).OnlyEnforceIf(greater_than_r1)
                model.Add(i < room.startRow).OnlyEnforceIf(
                    greater_than_r1.Not())
                # strictly less due to the mapping between continus and discrete systems
                model.Add(i < room.endRow).OnlyEnforceIf(less_than_r2)
                model.Add(i >= room.endRow).OnlyEnforceIf(less_than_r2.Not())
                # cols
                greater_than_c1 = model.NewBoolVar('')
                less_than_c2 = model.NewBoolVar('')
                model.Add(j >= room.startCol).OnlyEnforceIf(greater_than_c1)
                model.Add(j < room.startCol).OnlyEnforceIf(
                    greater_than_c1.Not())
                # strictly less due to the mapping between continus and discrete systems
                model.Add(j < room.endCol).OnlyEnforceIf(less_than_c2)
                model.Add(j >= room.endCol).OnlyEnforceIf(less_than_c2.Not())
                betweenRows = model.NewBoolVar('')
                model.AddBoolAnd([greater_than_r1, less_than_r2]
                                 ).OnlyEnforceIf(betweenRows)
                model.AddBoolOr([greater_than_r1.Not(), less_than_r2.Not()]
                                ).OnlyEnforceIf(betweenRows.Not())
                betweenCols = model.NewBoolVar('')
                model.AddBoolAnd([greater_than_c1, less_than_c2]
                                 ).OnlyEnforceIf(betweenCols)
                model.AddBoolOr([greater_than_c1.Not(), less_than_c2.Not()]
                                ).OnlyEnforceIf(betweenCols.Not())

                model.Add(grid[i][j] == index).OnlyEnforceIf(
                    [betweenRows, betweenCols])
                intersection = model.NewBoolVar('')
                model.Add(intersection == 1).OnlyEnforceIf(
                    [betweenRows, betweenCols])
                model.AddImplication(betweenCols.Not(), intersection.Not())
                model.AddImplication(betweenRows.Not(), intersection.Not())
                intersections.append(intersection)
            empty = model.NewBoolVar('')
            model.Add(sum(intersections) == 0).OnlyEnforceIf(empty)
            model.Add(sum(intersections) > 0).OnlyEnforceIf(empty.Not())
            model.Add(grid[i][j] == -1).OnlyEnforceIf(empty)
    return grid

# Takes in the grid, universal.


def GetSunReachability(grid):
    """A cell is reachable if it's reachable from any of the four directions."""
    reachable_from = [[[model.NewBoolVar('%i %i %i' % (i, j, k)) for k in range(
        len(DI))] for j in range(maxDim)] for i in range(maxDim)]
    reachable = [[model.NewBoolVar('') for j in range(maxDim)]
                 for i in range(maxDim)]
    top = 0
    bottom = 1
    left = 2
    right = 3
    isOpen = 1 if FLOOR_TOP_SIDE == BuildingSide.OPEN else 0
    for j in range(maxDim):
        model.Add(reachable_from[0][j][top] == isOpen)

    isOpen = 1 if FLOOR_BOTTOM_SIDE == BuildingSide.OPEN else 0
    for j in range(maxDim):
        model.Add(reachable_from[maxDim - 1][j][bottom] == isOpen)

    isOpen = 1 if FLOOR_LEFT_SIDE == BuildingSide.OPEN else 0
    for i in range(maxDim):
        model.Add(reachable_from[i][0][left] == isOpen)

    isOpen = 1 if FLOOR_RIGHT_SIDE == BuildingSide.OPEN else 0
    for i in range(maxDim):
        model.Add(reachable_from[i][maxDim-1][right] == isOpen)

    for i in range(maxDim):
        for j in range(maxDim):
            currentCell = []
            # if (neighbor ==1 and (equal to neighbor or neighbor is empty))
            for k in range(len(DI)):
                i2 = i+DI[k]
                j2 = j+DJ[k]
                if(i2 < 0 or i2 >= maxDim):
                    currentCell.append(reachable_from[i][j][k])
                    continue
                if(j2 < 0 or j2 >= maxDim):
                    currentCell.append(reachable_from[i][j][k])
                    continue
                equal = model.NewBoolVar('')
                model.Add(grid[i][j] == grid[i2][j2]).OnlyEnforceIf(equal)
                model.Add(grid[i][j] != grid[i2][j2]
                          ).OnlyEnforceIf(equal.Not())
                neighbor_empty = model.NewBoolVar('')
                model.Add(grid[i2][j2] == -1).OnlyEnforceIf(neighbor_empty)
                model.Add(grid[i2][j2] != -
                          1).OnlyEnforceIf(neighbor_empty.Not())
                notBlocked = model.NewBoolVar('')
                model.AddImplication(neighbor_empty, notBlocked)
                model.AddImplication(equal, notBlocked)
                model.Add(notBlocked == 0).OnlyEnforceIf(
                    [equal.Not(), neighbor_empty.Not()])
                model.Add(reachable_from[i][j][k] == 1).OnlyEnforceIf(
                    [reachable_from[i2][j2][k], notBlocked])
                model.AddImplication(
                    reachable_from[i2][j2][k].Not(), reachable_from[i][j][k].Not())
                model.AddImplication(
                    notBlocked.Not(), reachable_from[i][j][k].Not())
                currentCell.append(reachable_from[i][j][k])
            model.Add(sum(currentCell) > 0).OnlyEnforceIf(reachable[i][j])
            model.Add(sum(currentCell) == 0).OnlyEnforceIf(
                reachable[i][j].Not())
    return reachable

# Takes in the flattened version of the apartments, universal.


def AddSunRoomConstraints(sun_reachability, grid, flattened_rooms):
    """For each sunrom, one of its cells must be reachable from the sun."""
    for index, room in enumerate(flattened_rooms):
        if(room.room_type == Room.SUNROOM):
            isreachable = []
            for i in range(maxDim):
                for j in range(maxDim):
                    # if grid[i][j]==index and sun_reachability[i][j]==1 then true
                    b = model.NewBoolVar('')
                    inRoom = model.NewBoolVar('')
                    model.Add(grid[i][j] == index).OnlyEnforceIf(inRoom)
                    model.Add(grid[i][j] != index).OnlyEnforceIf(inRoom.Not())
                    model.Add(b == 1).OnlyEnforceIf(
                        [inRoom, sun_reachability[i][j]])
                    model.AddImplication(inRoom.Not(), b.Not())
                    model.AddImplication(sun_reachability[i][j].Not(), b.Not())
                    isreachable.append(b)
            model.Add(sum(isreachable) > 0)


def FlattenRooms(apartments):
    flattenedRooms = []
    for apartment in apartments:
        for room in apartment:
            flattenedRooms.append(room)
    return flattenedRooms

############ Debugging Functions ##################################


def CheckGrid(flattenedRooms, grid):
    """Checks that the creeated Int var grid is equal to the visualized grid."""
    visited = [[False for j in range(maxDim)] for i in range(maxDim)]
    for index, room in enumerate(flattenedRooms):
        r1 = solver.Value(room.startRow)
        r2 = solver.Value(room.endRow)
        c1 = solver.Value(room.startCol)
        c2 = solver.Value(room.endCol)
        for r in range(r1, r2):
            for c in range(c1, c2):
                curr = solver.Value(grid[r][c])
                assert(curr == index)
                visited[r][c] = True
    for i in range(maxDim):
        for j in range(maxDim):
            assert(visited[i][j] or solver.Value(grid[i][j]) == -1)


def PrintSunReachability(sun_reachibility):
    print('Sun Reachability Matrix')
    for i in range(maxDim):
        for j in range(maxDim):
            print(solver.Value(sun_reachability[i][j]), end=' ')
        print()


def PrintApartment(flattenedRooms):
    """Prints all maxDim * maxDim values in constrast to the visualize method which prints only the bounding box."""
    # print('Complete Apartment')
    visualizedOutput = [[0 for i in range(maxDim)] for j in range(maxDim)]
    fig, ax = plt.subplots()
    for i, row in enumerate(grid):
        for j, x in enumerate(row):
            value = solver.Value(x) + 1
            room_type = flattenedRooms[value - 1].room_type
            apartmentNum = flattenedRooms[value - 1].apartment
            if value == 0:
                room_type = Room.OTHER
                apartmentNum = ''
            print(value, end=' ')
            visualizedOutput[i][j] = value
            ax.text(
                j, i, ROOM_TYPE_MAP[str(room_type)]+','+str(apartmentNum), ha='center', va='center')
        print()

    im = ax.imshow(visualizedOutput)
    fig.tight_layout()
    plt.show()

########################   Main Method Starts Here   ########################

########################   Process Future Input Here ########################


n_apartments = 2

apartments = []


model = cp_model.CpModel()
for apartment_no in range(n_apartments):
    n_corridors = 3
    n_rooms = 6 + n_corridors
    min_area = [randint(1, 5) for i in range(n_rooms)]
    print(min_area)
    apartment = []
    for i in range(n_rooms):
        if i >= n_rooms - n_corridors:
            apartment.append(
                Rectangle(Room.CORRIDOR, apartment=apartment_no + 1))
            continue

        room_type = Room.OTHER
        adjacent_to = -1
        if i == 0:
            room_type = Room.KITCHEN
        elif i == 1:
            room_type = Room.DININGROOM
        elif i == 2:
            room_type = Room.MINOR_BATHROOM
            adjacent_to = 0
        elif i == 3:
            room_type = Room.BEDROOM
        elif i == 4:
            room_type = Room.DRESSING_ROOM
            adjacent_to = 3
        elif i == 5:
            room_type = Room.SUNROOM
        apartment.append(
            Rectangle(room_type, min_area[i], adjacent_to=adjacent_to, apartment=apartment_no + 1))

    apartments.append(apartment)

########################   Process Future Input Here ########################
for apartment in apartments:
    for room in apartment:
        room.addRoomConstraints(apartment)

flattened_rooms = FlattenRooms(apartments)

AddNoIntersectionConstraint(flattened_rooms)

for apartment in apartments:
    AddCorridorConstraint(n_corridors, apartment)

grid = GetGrid(flattened_rooms)
sun_reachability = GetSunReachability(grid)
AddSunRoomConstraints(sun_reachability, grid, flattened_rooms)
solver = cp_model.CpSolver()
status = solver.Solve(model)
print(solver.StatusName())
print('time = ', solver.WallTime())

########################   Main Method Ends Here   ##########################

########################  Debuging ################

CheckGrid(flattened_rooms, grid)

PrintApartment(flattened_rooms)
