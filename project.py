from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ortools.sat.python import cp_model

import matplotlib.pyplot as plt

from enum import Enum
from random import randint


########################   Enums   ########################

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

########################   Enums   ########################

########################   Global Variables   ########################


max_dim = 10

DI = [-1, 1, 0, 0]
DJ = [0, 0, -1, 1]

ROOM_TYPE_MAP = {'Room.DININGROOM': 'DR', 'Room.KITCHEN': 'KT', 'Room.MINOR_BATHROOM': 'MB',
                 'Room.DRESSING_ROOM': 'DRS', 'Room.BEDROOM': 'BD', 'Room.SUNROOM': 'SR', 'Room.CORRIDOR': 'C', 'Room.OTHER': 'X'}


FLOOR_RIGHT_SIDE = BuildingSide.OPEN
FLOOR_LEFT_SIDE = BuildingSide.LANDSCAPE
FLOOR_TOP_SIDE = BuildingSide.LANDSCAPE
FLOOR_BOTTOM_SIDE = BuildingSide.NONE

########################   Global Variables   ########################

########################   Classes   ########################


class Rectangle:
    roomId = 1

    def __init__(self, room_type, min_area=1, width=0, height=0, adjacent_to=-1, apartment=0):
        # Name the variable names in the model properly.
        self.width = model.NewIntVar(
            1, max_dim, 'Width, room: %d' % Rectangle.roomId)
        self.height = model.NewIntVar(
            1, max_dim, 'Height, room: %d' % Rectangle.roomId)
        self.area = model.NewIntVar(
            min_area, max_dim * max_dim, 'Area, room: %d' % Rectangle.roomId)
        self.start_row = model.NewIntVar(
            0, max_dim, 'Starting row, room: %d' % Rectangle.roomId)
        self.start_col = model.NewIntVar(
            0, max_dim, 'Starting col, room: %d' % Rectangle.roomId)
        self.end_row = model.NewIntVar(
            0, max_dim, 'Ending row, room: %d' % Rectangle.roomId)
        self.end_col = model.NewIntVar(
            0, max_dim, 'Ending col, room: %d' % Rectangle.roomId)
        self.room_type = room_type
        self.apartment = apartment

        self.add_generic_constraints(width, height)
        self.adjacent_to = adjacent_to
        Rectangle.roomId += 1

    def add_generic_constraints(self, width, height):
        # We call the methods if the method is invoked with these parameters.
        if(width > 0):
            self.add_width(width)
        if(height > 0):
            self.add_height(height)

        model.Add(self.width == self.end_col-self.start_col)
        model.Add(self.height == self.end_row-self.start_row)
        model.AddMultiplicationEquality(self.area, [self.width, self.height])

    def add_room_constraints(self, apartment):
        adjacent_to = self.adjacent_to
        if self.room_type == Room.DININGROOM:
            for i in range(len(apartment)):
                if(apartment[i].room_type == Room.KITCHEN):
                    adjacent_to = i
        if adjacent_to != -1:
            add_adjacency_constraint(self, apartment[adjacent_to])

    def room_exists_within_columns(self, startCol, endCol):
        return add_intersection_between_edges(
            [self.start_col, self.end_col], [startCol, endCol])

    def room_exists_within_rows(self, startRow, endRow):
        return add_intersection_between_edges(
            [self.start_row, self.end_row], [startRow, endRow])

    def to_string(self):
        print("Rectangle coordinates: (%d,%d)" %
              (self.start_row, self.start_col))
        print("Rectangle width: %d, Rectangle height: %d" %
              (self.width, self.height))

    def add_width(self, width):
        model.Add(self.width == width)

    def add_height(self, height):
        model.Add(self.height == height)

    def get_left(self):
        return self.start_col

    def get_right(self):
        return self.end_col

    def get_top(self):
        return self.start_row

    def get_bottom(self):
        return self.end_row

########################   Classes   ########################


def add_intersection_between_edges(a, b):
    l1 = a[0]
    r1 = a[1]
    l2 = b[0]
    r2 = b[1]
    eq = model.NewBoolVar('')
    l = model.NewIntVar(0, max_dim, '')
    model.AddMaxEquality(l, [l1, l2])
    r = model.NewIntVar(0, max_dim, '')
    model.AddMinEquality(r, [r1, r2])
    leq = model.NewBoolVar('')
    model.Add(l <= r).OnlyEnforceIf(leq)
    model.Add(l > r).OnlyEnforceIf(leq.Not())
    model.Add(l == r).OnlyEnforceIf(eq)
    model.Add(l != r).OnlyEnforceIf(eq.Not())
    return leq, eq

# This method sets the relation between the start and end (rows/columns)
# by adding the |AddNoOverlap2D| constraint to the model.

# Takes in the flattened version of the apartments, universal. Takes in all corridors.


def add_no_intersection_constraint(flattened_rooms, floor_corridors):
    row_intervals = [model.NewIntervalVar(
        room.get_top(), room.height, room.get_bottom(), 'room %d' % (roomNum + 1)) for roomNum, room in enumerate(flattened_rooms)]
    col_intervals = [model.NewIntervalVar(
        room.get_left(), room.width, room.get_right(), 'room %d' % (roomNum + 1)) for roomNum, room in enumerate(flattened_rooms)]
    for corridor in floor_corridors:
        row_intervals.append(model.NewIntervalVar(
            corridor.get_top(), corridor.height, corridor.get_bottom(), ''))
        col_intervals.append(model.NewIntervalVar(
            corridor.get_left(), corridor.width, corridor.get_right(), ''))

    model.AddNoOverlap2D(col_intervals, row_intervals)


def add_adjacency_constraint(room, adjacent_room, add=1):
    columns_leq, columns_eq = room.room_exists_within_columns(
        adjacent_room.start_col, adjacent_room.end_col)
    rows_leq, rows_eq = room.room_exists_within_rows(
        adjacent_room.start_row, adjacent_room.end_row)
    intersection = model.NewBoolVar('')
    model.Add(intersection == 1).OnlyEnforceIf([columns_leq, rows_leq])
    model.AddImplication(columns_leq.Not(), intersection.Not())
    model.AddImplication(rows_leq.Not(), intersection.Not())
    if add == 1:
        model.Add(intersection == 1)
    model.Add(columns_eq + rows_eq < 2)
    return intersection

# Takes in one single apartment, non-universal.


def add_corridor_constraint(n_corridors, flattened_rooms):
    '''The last nOfCorriodors should have type corridor'''
    assert(n_corridors > 0)
    n = len(flattened_rooms)
    # All the corriods are adjacent to each other
    for i in range(n-n_corridors, n-1):
        add_adjacency_constraint(flattened_rooms[i], flattened_rooms[i+1])
    for i in range(n-n_corridors):
        current_room = flattened_rooms[i]
        adjacent_to_corridors = []
        for j in range(n-n_corridors, n):
            corridor = flattened_rooms[j]
            adjacent_to_corridors.append(add_adjacency_constraint(
                current_room, corridor, 0))
        model.Add(sum(adjacent_to_corridors) > 0)


def add_floor_corridor_constraints(apartments, floor_corridors):
    '''There has to be at least one corridor for the floor'''
    assert(len(floor_corridors) > 0)

    n_floor_corridors = len(floor_corridors)
    # All floor corridors are adjacent to each other
    for i in range(n_floor_corridors - 1):
        add_adjacency_constraint(floor_corridors[i], floor_corridors[i + 1])

    # At least one room from each apartment is adjacent to a corridor
    for apartment in apartments:
        adjacent_to_corridors = []
        for room in apartment:
            for corridor in floor_corridors:
                adjacent_to_corridors.append(
                    add_adjacency_constraint(room, corridor, 0))
        model.Add(sum(adjacent_to_corridors) > 0)

# Takes in the flattened version of the apartments, universal.
# Consider corridors. For now it takes in all corridors.


def get_grid(flattened_floor):
    n = len(flattened_floor)
    grid = [[model.NewIntVar(-1, n-1, '') for j in range(max_dim)]
            for i in range(max_dim)]
    for i in range(max_dim):
        for j in range(max_dim):
            intersections = []
            for index, room in enumerate(flattened_floor):

                # rows
                greater_than_r1 = model.NewBoolVar('')
                less_than_r2 = model.NewBoolVar('')
                model.Add(i >= room.start_row).OnlyEnforceIf(greater_than_r1)
                model.Add(i < room.start_row).OnlyEnforceIf(
                    greater_than_r1.Not())
                # strictly less due to the mapping between continus and discrete systems
                model.Add(i < room.end_row).OnlyEnforceIf(less_than_r2)
                model.Add(i >= room.end_row).OnlyEnforceIf(less_than_r2.Not())
                # cols
                greater_than_c1 = model.NewBoolVar('')
                less_than_c2 = model.NewBoolVar('')
                model.Add(j >= room.start_col).OnlyEnforceIf(greater_than_c1)
                model.Add(j < room.start_col).OnlyEnforceIf(
                    greater_than_c1.Not())
                # strictly less due to the mapping between continus and discrete systems
                model.Add(j < room.end_col).OnlyEnforceIf(less_than_c2)
                model.Add(j >= room.end_col).OnlyEnforceIf(less_than_c2.Not())
                between_rows = model.NewBoolVar('')
                model.AddBoolAnd([greater_than_r1, less_than_r2]
                                 ).OnlyEnforceIf(between_rows)
                model.AddBoolOr([greater_than_r1.Not(), less_than_r2.Not()]
                                ).OnlyEnforceIf(between_rows.Not())
                between_columns = model.NewBoolVar('')
                model.AddBoolAnd([greater_than_c1, less_than_c2]
                                 ).OnlyEnforceIf(between_columns)
                model.AddBoolOr([greater_than_c1.Not(), less_than_c2.Not()]
                                ).OnlyEnforceIf(between_columns.Not())

                model.Add(grid[i][j] == index).OnlyEnforceIf(
                    [between_rows, between_columns])
                intersection = model.NewBoolVar('')
                model.Add(intersection == 1).OnlyEnforceIf(
                    [between_rows, between_columns])
                model.AddImplication(between_columns.Not(), intersection.Not())
                model.AddImplication(between_rows.Not(), intersection.Not())
                intersections.append(intersection)
            empty = model.NewBoolVar('')
            model.Add(sum(intersections) == 0).OnlyEnforceIf(empty)
            model.Add(sum(intersections) > 0).OnlyEnforceIf(empty.Not())
            model.Add(grid[i][j] == -1).OnlyEnforceIf(empty)
    return grid

# Takes in the grid, universal.
# Consider corridors as well.


def get_sun_reachability(grid):
    """A cell is reachable if it's reachable from any of the four directions."""
    reachable_from = [[[model.NewBoolVar('%i %i %i' % (i, j, k)) for k in range(
        len(DI))] for j in range(max_dim)] for i in range(max_dim)]
    reachable = [[model.NewBoolVar('') for j in range(max_dim)]
                 for i in range(max_dim)]
    top = 0
    bottom = 1
    left = 2
    right = 3
    is_open = 1 if FLOOR_TOP_SIDE == BuildingSide.OPEN else 0

    for j in range(max_dim):
        model.Add(reachable_from[0][j][top] == is_open)

    is_open = 1 if FLOOR_BOTTOM_SIDE == BuildingSide.OPEN else 0
    for j in range(max_dim):
        model.Add(reachable_from[max_dim - 1][j][bottom] == is_open)

    is_open = 1 if FLOOR_LEFT_SIDE == BuildingSide.OPEN else 0
    for i in range(max_dim):
        model.Add(reachable_from[i][0][left] == is_open)

    is_open = 1 if FLOOR_RIGHT_SIDE == BuildingSide.OPEN else 0
    for i in range(max_dim):
        model.Add(reachable_from[i][max_dim-1][right] == is_open)

    for i in range(max_dim):
        for j in range(max_dim):
            current_cell = []
            # if (neighbor ==1 and (equal to neighbor or neighbor is empty))
            for k in range(len(DI)):
                i2 = i+DI[k]
                j2 = j+DJ[k]
                if(i2 < 0 or i2 >= max_dim):
                    current_cell.append(reachable_from[i][j][k])
                    continue
                if(j2 < 0 or j2 >= max_dim):
                    current_cell.append(reachable_from[i][j][k])
                    continue
                equal = model.NewBoolVar('')
                model.Add(grid[i][j] == grid[i2][j2]).OnlyEnforceIf(equal)
                model.Add(grid[i][j] != grid[i2][j2]
                          ).OnlyEnforceIf(equal.Not())
                neighbor_empty = model.NewBoolVar('')
                model.Add(grid[i2][j2] == -1).OnlyEnforceIf(neighbor_empty)
                model.Add(grid[i2][j2] != -
                          1).OnlyEnforceIf(neighbor_empty.Not())
                not_blocked = model.NewBoolVar('')
                model.AddImplication(neighbor_empty, not_blocked)
                model.AddImplication(equal, not_blocked)
                model.Add(not_blocked == 0).OnlyEnforceIf(
                    [equal.Not(), neighbor_empty.Not()])
                model.Add(reachable_from[i][j][k] == 1).OnlyEnforceIf(
                    [reachable_from[i2][j2][k], not_blocked])
                model.AddImplication(
                    reachable_from[i2][j2][k].Not(), reachable_from[i][j][k].Not())
                model.AddImplication(
                    not_blocked.Not(), reachable_from[i][j][k].Not())
                current_cell.append(reachable_from[i][j][k])
            model.Add(sum(current_cell) > 0).OnlyEnforceIf(reachable[i][j])
            model.Add(sum(current_cell) == 0).OnlyEnforceIf(
                reachable[i][j].Not())
    return reachable

# Takes in the flattened version of the apartments, universal.


def add_sunroom_constraints(sun_reachability, grid, flattened_rooms):
    """For each sunrom, one of its cells must be reachable from the sun."""
    for index, room in enumerate(flattened_rooms):
        if(room.room_type == Room.SUNROOM):
            is_reachable = []
            for i in range(max_dim):
                for j in range(max_dim):
                    # if grid[i][j]==index and sun_reachability[i][j]==1 then true
                    b = model.NewBoolVar('')
                    in_room = model.NewBoolVar('')
                    model.Add(grid[i][j] == index).OnlyEnforceIf(in_room)
                    model.Add(grid[i][j] != index).OnlyEnforceIf(in_room.Not())
                    model.Add(b == 1).OnlyEnforceIf(
                        [in_room, sun_reachability[i][j]])
                    model.AddImplication(in_room.Not(), b.Not())
                    model.AddImplication(sun_reachability[i][j].Not(), b.Not())
                    is_reachable.append(b)
            model.Add(sum(is_reachable) > 0)

# If given |apartments| only, it will flatten the apartments.
# If given |apartments| and |floor_corridors| it will flatten both together.


def flatten(apartments, floor_corridors=[]):
    flattened_rooms = []
    for apartment in apartments:
        for room in apartment:
            flattened_rooms.append(room)

    for corridor in floor_corridors:
        flattened_rooms.append(corridor)
    return flattened_rooms

############ Debugging Functions ##################################


def check_grid(flattened_floor, grid):
    """Checks that the creeated Int var grid is equal to the visualized grid."""
    visited = [[False for j in range(max_dim)] for i in range(max_dim)]
    for index, room in enumerate(flattened_floor):
        start_row = solver.Value(room.start_row)
        end_row = solver.Value(room.end_row)
        start_column = solver.Value(room.start_col)
        end_column = solver.Value(room.end_col)
        for row in range(start_row, end_row):
            for column in range(start_column, end_column):
                curr = solver.Value(grid[row][column])
                assert(curr == index)
                visited[row][column] = True
    for i in range(max_dim):
        for j in range(max_dim):
            assert(visited[i][j] or solver.Value(grid[i][j]) == -1)


def print_sun_reachability(sun_reachibility):
    print('Sun Reachability Matrix')
    for i in range(max_dim):
        for j in range(max_dim):
            print(solver.Value(sun_reachability[i][j]), end=' ')
        print()


def visualize_floor(flattened_floor):
    """Visualizes the floor using matplotlib"""
    visualized_output = [[0 for i in range(max_dim)] for j in range(max_dim)]
    fig, ax = plt.subplots()
    for i, row in enumerate(grid):
        for j, x in enumerate(row):
            value = solver.Value(x) + 1
            room_type = flattened_floor[value - 1].room_type
            apartment_num = flattened_floor[value - 1].apartment
            heatmap_text = ROOM_TYPE_MAP[str(room_type)]+','+str(apartment_num)
            if value == 0:
                room_type = Room.OTHER
                apartment_num = ''
                heatmap_text = ROOM_TYPE_MAP[str(room_type)]
            visualized_output[i][j] = value
            ax.text(
                j, i, heatmap_text, ha='center', va='center')

    im = ax.imshow(visualized_output)
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

n_floor_corridors = 3
floor_corridors = []
for i in range(n_floor_corridors):
    floor_corridors.append(Rectangle(Room.CORRIDOR))

########################   Process Future Input Here ########################
for apartment in apartments:
    for room in apartment:
        room.add_room_constraints(apartment)

flattened_rooms = flatten(apartments)
flattened_floor = flatten(apartments, floor_corridors)

add_no_intersection_constraint(flattened_rooms, floor_corridors)
add_floor_corridor_constraints(apartments, floor_corridors)

for apartment in apartments:
    add_corridor_constraint(n_corridors, apartment)

grid = get_grid(flattened_floor)
sun_reachability = get_sun_reachability(grid)
add_sunroom_constraints(sun_reachability, grid, flattened_rooms)
solver = cp_model.CpSolver()
status = solver.Solve(model)
print(solver.StatusName())
print('time = ', solver.WallTime())

########################   Main Method Ends Here   ##########################

########################  Debuging ################

check_grid(flattened_floor, grid)

visualize_floor(flattened_floor)
