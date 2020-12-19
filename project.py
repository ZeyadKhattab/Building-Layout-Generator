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
    MAIN_BATHROOM = 3
    MINOR_BATHROOM = 4
    DUCT = 5
    DRESSING_ROOM = 6
    BEDROOM = 7
    SUNROOM = 8
    CORRIDOR = 9
    STAIR = 10
    ELEVATOR = 11
    OTHER = 12


class FloorSide(Enum):
    LANDSCAPE = 1
    OPEN = 2
    NONE = 3

########################   Enums   ########################

########################   Global Variables   ########################


MAX_DIM = 10

DI = [-1, 1, 0, 0]
DJ = [0, 0, -1, 1]

ROOM_TYPE_MAP = {'Room.DININGROOM': 'DR', 'Room.KITCHEN': 'KT', 'Room.MAIN_BATHROOM': 'MB', 'Room.MINOR_BATHROOM': 'mb',
                 'Room.DRESSING_ROOM': 'DRS', 'Room.BEDROOM': 'BD', 'Room.SUNROOM': 'SR', 'Room.CORRIDOR': 'C',
                 'Room.DUCT': 'D', 'Room.STAIR': 'S', 'Room.ELEVATOR': 'E', 'Room.OTHER': 'X'}


FLOOR_RIGHT_SIDE = FloorSide.OPEN
FLOOR_LEFT_SIDE = FloorSide.LANDSCAPE
FLOOR_TOP_SIDE = FloorSide.LANDSCAPE
FLOOR_BOTTOM_SIDE = FloorSide.NONE

########################   Global Variables   ########################

########################   Classes   ########################


class Rectangle:
    room_id = 1

    def __init__(self, room_type, min_area=1, width=0, height=0, adjacent_to=-1, apartment=0):
        # Name the variable names in the model properly.
        self.width = model.NewIntVar(
            1, MAX_DIM, 'Width, room: %d' % Rectangle.room_id)
        self.height = model.NewIntVar(
            1, MAX_DIM, 'Height, room: %d' % Rectangle.room_id)
        self.area = model.NewIntVar(
            min_area, MAX_DIM * MAX_DIM, 'Area, room: %d' % Rectangle.room_id)
        self.start_row = model.NewIntVar(
            0, MAX_DIM, 'Starting row, room: %d' % Rectangle.room_id)
        self.start_col = model.NewIntVar(
            0, MAX_DIM, 'Starting col, room: %d' % Rectangle.room_id)
        self.end_row = model.NewIntVar(
            0, MAX_DIM, 'Ending row, room: %d' % Rectangle.room_id)
        self.end_col = model.NewIntVar(
            0, MAX_DIM, 'Ending col, room: %d' % Rectangle.room_id)
        self.room_type = room_type
        self.apartment = apartment

        self.add_generic_constraints(width, height)
        self.adjacent_to = adjacent_to
        Rectangle.room_id += 1

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


def add_intersection_between_edges(edge_a, edge_b):
    start_edge_a = edge_a[0]
    end_edge_a = edge_a[1]
    start_edge_b = edge_b[0]
    end_edge_b = edge_b[1]
    eq = model.NewBoolVar('')
    max_edge_start = model.NewIntVar(0, MAX_DIM, '')
    model.AddMaxEquality(max_edge_start, [start_edge_a, start_edge_b])
    min_edge_end = model.NewIntVar(0, MAX_DIM, '')
    model.AddMinEquality(min_edge_end, [end_edge_a, end_edge_b])
    leq = model.NewBoolVar('')
    model.Add(max_edge_start <= min_edge_end).OnlyEnforceIf(leq)
    model.Add(max_edge_start > min_edge_end).OnlyEnforceIf(leq.Not())
    model.Add(max_edge_start == min_edge_end).OnlyEnforceIf(eq)
    model.Add(max_edge_start != min_edge_end).OnlyEnforceIf(eq.Not())
    return leq, eq

# This method sets the relation between the start and end (rows/columns)
# by adding the |AddNoOverlap2D| constraint to the model.

# Takes in the flattened floor.


def add_no_intersection_constraint(flattened_floor):
    row_intervals = [model.NewIntervalVar(
        room.get_top(), room.height, room.get_bottom(), 'room %d' % (roomNum + 1)) for roomNum, room in enumerate(flattened_floor)]
    col_intervals = [model.NewIntervalVar(
        room.get_left(), room.width, room.get_right(), 'room %d' % (roomNum + 1)) for roomNum, room in enumerate(flattened_floor)]

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


def add_corridor_constraint(n_corridors, apartment):
    '''The last nOfCorriodors should have type corridor'''
    assert(n_corridors > 0)
    n_rooms = len(apartment)
    # All the corriods are adjacent to each other
    for i in range(n_rooms-n_corridors, n_rooms-1):
        add_adjacency_constraint(apartment[i], apartment[i+1])
    for i in range(n_rooms-n_corridors):
        current_room = apartment[i]
        adjacent_to_corridors = []
        for j in range(n_rooms-n_corridors, n_rooms):
            corridor = apartment[j]
            adjacent_to_corridors.append(add_adjacency_constraint(
                current_room, corridor, 0))
        model.Add(sum(adjacent_to_corridors) > 0)


def add_duct_constraints(apartment_ducts, apartment):
    assert(len(apartment_ducts) > 0)

    for room in apartment:
        if room.room_type == Room.KITCHEN or room.room_type == Room.MINOR_BATHROOM or room.room_type == Room.MAIN_BATHROOM:
            adjacent_to_ducts = []
            for duct in apartment_ducts:
                adjacent_to_ducts.append(
                    add_adjacency_constraint(room, duct, 0))
            model.Add(sum(adjacent_to_ducts) > 0)

    for duct in apartment_ducts:
        duct_adjacent_to = []
        for room in apartment:
            if room.room_type == Room.KITCHEN or room.room_type == Room.MINOR_BATHROOM or room.room_type == Room.MAIN_BATHROOM:
                duct_adjacent_to.append(
                    add_adjacency_constraint(duct, room, 0))
        model.Add(sum(duct_adjacent_to) > 0)


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


def add_stair_elevator_constraints(stair, elevator, floor_corridors):
    add_adjacency_constraint(stair, floor_corridors[0])
    add_adjacency_constraint(elevator, floor_corridors[0])

# Takes in the flattened version of the apartments, universal.
# Consider corridors. For now it takes in all corridors.


def get_grid(flattened_floor):
    n = len(flattened_floor)
    grid = [[model.NewIntVar(-1, n-1, '') for j in range(MAX_DIM)]
            for i in range(MAX_DIM)]
    for i in range(MAX_DIM):
        for j in range(MAX_DIM):
            intersections = []
            for index, room in enumerate(flattened_floor):

                # rows
                greater_than_start_row = model.NewBoolVar('')
                less_than_end_row = model.NewBoolVar('')
                model.Add(i >= room.start_row).OnlyEnforceIf(
                    greater_than_start_row)
                model.Add(i < room.start_row).OnlyEnforceIf(
                    greater_than_start_row.Not())
                # strictly less due to the mapping between continus and discrete systems
                model.Add(i < room.end_row).OnlyEnforceIf(less_than_end_row)
                model.Add(i >= room.end_row).OnlyEnforceIf(
                    less_than_end_row.Not())
                # cols
                greater_than_start_col = model.NewBoolVar('')
                less_than_end_col = model.NewBoolVar('')
                model.Add(j >= room.start_col).OnlyEnforceIf(
                    greater_than_start_col)
                model.Add(j < room.start_col).OnlyEnforceIf(
                    greater_than_start_col.Not())
                # strictly less due to the mapping between continus and discrete systems
                model.Add(j < room.end_col).OnlyEnforceIf(less_than_end_col)
                model.Add(j >= room.end_col).OnlyEnforceIf(
                    less_than_end_col.Not())
                between_rows = model.NewBoolVar('')
                model.AddBoolAnd([greater_than_start_row, less_than_end_row]
                                 ).OnlyEnforceIf(between_rows)
                model.AddBoolOr([greater_than_start_row.Not(), less_than_end_row.Not()]
                                ).OnlyEnforceIf(between_rows.Not())
                between_columns = model.NewBoolVar('')
                model.AddBoolAnd([greater_than_start_col, less_than_end_col]
                                 ).OnlyEnforceIf(between_columns)
                model.AddBoolOr([greater_than_start_col.Not(), less_than_end_col.Not()]
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
        len(DI))] for j in range(MAX_DIM)] for i in range(MAX_DIM)]
    reachable = [[model.NewBoolVar('') for j in range(MAX_DIM)]
                 for i in range(MAX_DIM)]
    top = 0
    bottom = 1
    left = 2
    right = 3
    is_open = 1 if FLOOR_TOP_SIDE == FloorSide.OPEN else 0

    for j in range(MAX_DIM):
        model.Add(reachable_from[0][j][top] == is_open)

    is_open = 1 if FLOOR_BOTTOM_SIDE == FloorSide.OPEN else 0
    for j in range(MAX_DIM):
        model.Add(reachable_from[MAX_DIM - 1][j][bottom] == is_open)

    is_open = 1 if FLOOR_LEFT_SIDE == FloorSide.OPEN else 0
    for i in range(MAX_DIM):
        model.Add(reachable_from[i][0][left] == is_open)

    is_open = 1 if FLOOR_RIGHT_SIDE == FloorSide.OPEN else 0
    for i in range(MAX_DIM):
        model.Add(reachable_from[i][MAX_DIM-1][right] == is_open)

    for i in range(MAX_DIM):
        for j in range(MAX_DIM):
            current_cell = []
            # if (neighbor == 1 and (equal to neighbor or neighbor is empty))
            for k in range(len(DI)):
                i2 = i+DI[k]
                j2 = j+DJ[k]
                if(i2 < 0 or i2 >= MAX_DIM):
                    current_cell.append(reachable_from[i][j][k])
                    continue
                if(j2 < 0 or j2 >= MAX_DIM):
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


def add_sun_reachability(sun_reachability, grid, flattened_floor, room_idx):
    """For each sunrom, one of its cells must be reachable from the sun."""

    is_cells_reachable = []
    for i in range(MAX_DIM):
        for j in range(MAX_DIM):
            # if grid[i][j]==index and sun_reachability[i][j]==1 then true
            is_cell_reachable = model.NewBoolVar('')
            in_room = model.NewBoolVar('')
            model.Add(grid[i][j] == room_idx).OnlyEnforceIf(in_room)
            model.Add(grid[i][j] != room_idx).OnlyEnforceIf(in_room.Not())
            model.Add(is_cell_reachable == 1).OnlyEnforceIf(
                [in_room, sun_reachability[i][j]])
            model.AddImplication(
                in_room.Not(), is_cell_reachable.Not())
            model.AddImplication(
                sun_reachability[i][j].Not(), is_cell_reachable.Not())
            is_cells_reachable.append(is_cell_reachable)

    is_room_reachable = model.NewBoolVar('')
    model.Add(sum(is_cells_reachable) > 0).OnlyEnforceIf(is_room_reachable)
    model.Add(sum(is_cells_reachable) == 0).OnlyEnforceIf(
        is_room_reachable.Not())

    if flattened_floor[room_idx].room_type == Room.SUNROOM:
        model.Add(is_room_reachable == 1)

    return is_room_reachable


def add_soft_sun_reachability_constraint(sun_reachability, grid, flattened_floor):
    is_room_sun_reachable = []

    for room_idx, room in enumerate(flattened_floor):
        if room.room_type != Room.DUCT and room.room_type != Room.STAIR and room.room_type != Room.ELEVATOR and room.room_type != Room.SUNROOM:
            is_room_sun_reachable.append(add_sun_reachability(sun_reachability, grid,
                                                              flattened_floor, room_idx))

    return sum(is_room_sun_reachable)


def add_sunroom_constraint(sun_reachability, grid, flattened_floor):
    for room_idx, room in enumerate(flattened_floor):
        if room.room_type == Room.SUNROOM:
            add_sun_reachability(sun_reachability, grid,
                                 flattened_floor, room_idx)

# If given |apartments| only, it will flatten the apartments.
# If given |apartments| and |floor_corridors| it will flatten both together.


def flatten_floor(apartments, apartments_ducts, floor_corridors, stair, elevator):
    flattened_floor = []

    for apartment in apartments:
        for room in apartment:
            flattened_floor.append(room)

    for corridor in floor_corridors:
        flattened_floor.append(corridor)

    for apartment_ducts in apartments_ducts:
        for duct in apartment_ducts:
            flattened_floor.append(duct)

    flattened_floor.append(stair)
    flattened_floor.append(elevator)

    return flattened_floor

############ Debugging Functions ##################################


def check_grid(flattened_floor, grid):
    """Checks that the creeated Int var grid is equal to the visualized grid."""
    visited = [[False for j in range(MAX_DIM)] for i in range(MAX_DIM)]
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
    for i in range(MAX_DIM):
        for j in range(MAX_DIM):
            assert(visited[i][j] or solver.Value(grid[i][j]) == -1)


def print_sun_reachability(sun_reachibility):
    print('Sun Reachability Matrix')
    for i in range(MAX_DIM):
        for j in range(MAX_DIM):
            print(solver.Value(sun_reachability[i][j]), end=' ')
        print()


def visualize_floor(flattened_floor, grid):
    """Visualizes the floor using matplotlib"""
    visualized_output = [[0 for i in range(MAX_DIM)] for j in range(MAX_DIM)]
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
apartment_corridors = []
apartments_ducts = []

model = cp_model.CpModel()
for apartment_no in range(n_apartments):
    n_corridors = randint(1, 3)
    n_rooms = 6 + n_corridors
    apartment_corridors.append(n_corridors)
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

    n_ducts = randint(1, 4)
    apartment_ducts = []
    for duct_no in range(n_ducts):
        apartment_ducts.append(
            Rectangle(Room.DUCT, apartment=apartment_no + 1))
    apartments_ducts.append(apartment_ducts)

stair = Rectangle(Room.STAIR, width=1, height=1)
elevator = Rectangle(Room.ELEVATOR, width=1, height=1)

n_floor_corridors = randint(1, 3)
floor_corridors = []
for i in range(n_floor_corridors):
    floor_corridors.append(Rectangle(Room.CORRIDOR))

########################   Process Future Input Here ########################
for apartment in apartments:
    for room in apartment:
        room.add_room_constraints(apartment)

flattened_floor = flatten_floor(
    apartments, apartments_ducts, floor_corridors, stair, elevator)

add_no_intersection_constraint(flattened_floor)
add_floor_corridor_constraints(apartments, floor_corridors)
add_stair_elevator_constraints(stair, elevator, floor_corridors)

for apartment_no, apartment in enumerate(apartments):
    add_corridor_constraint(apartment_corridors[apartment_no], apartment)
    add_duct_constraints(apartments_ducts[apartment_no], apartment)

grid = get_grid(flattened_floor)
sun_reachability = get_sun_reachability(grid)
add_sunroom_constraint(sun_reachability, grid, flattened_floor)

model.Maximize(add_soft_sun_reachability_constraint(
    sun_reachability, grid, flattened_floor) * 1)

solver = cp_model.CpSolver()
status = solver.Solve(model)
print(solver.StatusName())
print('time = ', solver.WallTime())

########################   Main Method Ends Here   ##########################

########################  Debuging ################

check_grid(flattened_floor, grid)

visualize_floor(flattened_floor, grid)
