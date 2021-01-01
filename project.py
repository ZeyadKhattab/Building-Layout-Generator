from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ortools.sat.python import cp_model

import matplotlib.pyplot as plt

from enum import Enum
from random import randint


class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, flattened_floor, grid, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__flattened_floor = flattened_floor
        self.__grid = grid
        self.__solution_count = 0
        self.__solution_limit = limit

    def on_solution_callback(self):
        self.__solution_count += 1
        visualize_floor(self.__flattened_floor, self.__grid, self)
        if self.__solution_count >= self.__solution_limit:
            print('Stop search after %i solutions' % self.__solution_limit)
            self.StopSearch()

    def solution_count(self):
        return self.__solution_count
########################   Enums   ########################


class Equality(Enum):
    GREATER_THAN = 0
    LESS_THAN = 1
    NONE = 2


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
    LIVING_ROOM = 12
    OTHER = 13


class FloorSide(Enum):
    LANDSCAPE = 1
    OPEN = 2
    NONE = 3

########################   Enums   ########################

########################   Global Variables   ########################


DI = [-1, 1, 0, 0]
DJ = [0, 0, -1, 1]

IGNORE_REACHABILITY = [Room.CORRIDOR]
ROOM_TYPE_MAP = {'Room.DININGROOM': 'DR', 'Room.KITCHEN': 'KT', 'Room.MAIN_BATHROOM': 'MB', 'Room.MINOR_BATHROOM': 'mb',
                 'Room.DRESSING_ROOM': 'DRS', 'Room.BEDROOM': 'BD', 'Room.SUNROOM': 'SR', 'Room.CORRIDOR': 'C',
                 'Room.DUCT': 'D', 'Room.STAIR': 'S', 'Room.ELEVATOR': 'E', 'Room.LIVING_ROOM': 'LR', 'Room.OTHER': 'X'}


########################   Global Variables   ########################

########################   Classes   ########################


class Rectangle:
    room_id = 1

    def __init__(self, room_type, min_area=1, width=0, height=0, adjacent_to=-1, apartment=-1):
        # Name the variable names in the model properly.
        self.width = model.NewIntVar(
            1, FLOOR_WIDTH, 'Width, room: %d' % Rectangle.room_id)
        self.height = model.NewIntVar(
            1, FLOOR_LENGTH, 'Height, room: %d' % Rectangle.room_id)
        self.area = model.NewIntVar(
            min_area, FLOOR_LENGTH * FLOOR_WIDTH, 'Area, room: %d' % Rectangle.room_id)
        self.start_row = model.NewIntVar(
            0, FLOOR_LENGTH, 'Starting row, room: %d' % Rectangle.room_id)
        self.start_col = model.NewIntVar(
            0, FLOOR_WIDTH, 'Starting col, room: %d' % Rectangle.room_id)
        self.end_row = model.NewIntVar(
            0, FLOOR_LENGTH, 'Ending row, room: %d' % Rectangle.room_id)
        self.end_col = model.NewIntVar(
            0, FLOOR_WIDTH, 'Ending col, room: %d' % Rectangle.room_id)
        self.room_type = room_type
        self.apartment = apartment + 1

        self.add_generic_constraints(width, height)
        self.adjacent_to = adjacent_to
        Rectangle.room_id += 1

    def add_generic_constraints(self, width, height):
        # We call the methods if the method is invoked with these parameters.
        if(width > 0):
            # print('width in ctor', width)
            # print('add_width')
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

    def distance(self, other):
        left = model.NewBoolVar('')
        model.Add(other.end_col < self.start_col).OnlyEnforceIf(left)
        model.Add(other.end_col >= self.start_col).OnlyEnforceIf(left.Not())
        right = model.NewBoolVar('')
        model.Add(self.end_col < other.start_col).OnlyEnforceIf(right)
        model.Add(self.end_col >= other.start_col).OnlyEnforceIf(right.Not())

        bottom = model.NewBoolVar('')
        model.Add(other.start_row > self.end_row).OnlyEnforceIf(bottom)
        model.Add(other.start_row <= self.end_row).OnlyEnforceIf(bottom.Not())

        top = model.NewBoolVar('')
        model.Add(other.end_row < self.start_row).OnlyEnforceIf(top)
        model.Add(other.end_row >= self.start_row).OnlyEnforceIf(top.Not())

        dist = model.NewIntVar(0, FLOOR_LENGTH + FLOOR_WIDTH, '')

        model.Add(dist == (self.start_col - other.end_col) +
                  (self.start_row - other.end_row)).OnlyEnforceIf([top, left])

        model.Add(dist == (self.start_col - other.end_col) +
                  (other.start_row - self.end_row)).OnlyEnforceIf([bottom, left])

        model.Add(dist == other.start_row-self.end_row+other.start_col -
                  self.end_col).OnlyEnforceIf([bottom, right])
        model.Add(dist == self.start_row-other.end_row+other.start_col -
                  self.end_col).OnlyEnforceIf([right, top])

        top_left = model.NewBoolVar('')
        model.Add(top_left == 1).OnlyEnforceIf([top, left])
        model.AddImplication(top.Not(), top_left.Not())
        model.AddImplication(left.Not(), top_left.Not())

        bottom_left = model.NewBoolVar('')
        model.Add(bottom_left == 1).OnlyEnforceIf([bottom, left])
        model.AddImplication(bottom.Not(), bottom_left.Not())
        model.AddImplication(left.Not(), bottom_left.Not())

        top_right = model.NewBoolVar('')
        model.Add(top_right == 1).OnlyEnforceIf([top, right])
        model.AddImplication(top.Not(), top_right.Not())
        model.AddImplication(right.Not(), top_right.Not())

        bottom_right = model.NewBoolVar('')
        model.Add(bottom_right == 1).OnlyEnforceIf([bottom, right])
        model.AddImplication(bottom.Not(), bottom_right.Not())
        model.AddImplication(right.Not(), bottom_right.Not())

        model.Add(dist == self.start_col - other.end_col).OnlyEnforceIf(
            [left, bottom_right.Not(), bottom_left.Not(), top_right.Not(), top_left.Not()])

        model.Add(dist == other.start_col - self.end_col).OnlyEnforceIf(
            [right, bottom_right.Not(), bottom_left.Not(), top_right.Not(), top_left.Not()])

        model.Add(dist == self.start_row - other.end_row).OnlyEnforceIf(
            [top, bottom_right.Not(), bottom_left.Not(), top_right.Not(), top_left.Not()])

        model.Add(dist == other.start_row - self.end_row).OnlyEnforceIf(
            [bottom, bottom_right.Not(), bottom_left.Not(), top_right.Not(), top_left.Not()])

        model.Add(dist == 0).OnlyEnforceIf(
            [top.Not(), bottom.Not(), left.Not(), right.Not()])

        return dist
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
    for room_no in range(len(apartment) - n_corridors, len(apartment)):
        assert(apartment[room_no].room_type == Room.CORRIDOR)

    n_rooms = len(apartment)
    # All the corriods are adjacent to each other
    main_corridor = apartment[n_rooms-n_corridors]
    for i in range(n_rooms-n_corridors + 1, n_rooms):
        add_adjacency_constraint(apartment[i], main_corridor)
    for i in range(n_rooms-n_corridors):
        current_room = apartment[i]
        adjacent_to_corridors = []
        for j in range(n_rooms-n_corridors, n_rooms):
            corridor = apartment[j]
            adjacent_to_corridors.append(add_adjacency_constraint(
                current_room, corridor, 0))
        model.Add(sum(adjacent_to_corridors) > 0)


def add_duct_constraints(ducts, flattened_floor):
    assert(len(ducts) > 0)

    for room in flattened_floor:
        if room.room_type == Room.KITCHEN or room.room_type == Room.MINOR_BATHROOM or room.room_type == Room.MAIN_BATHROOM:
            adjacent_to_ducts = []
            for duct in ducts:
                adjacent_to_ducts.append(
                    add_adjacency_constraint(room, duct, 0))
            model.Add(sum(adjacent_to_ducts) > 0)

    for duct in ducts:
        duct_adjacent_to = []
        for room in flattened_floor:
            if room.room_type == Room.KITCHEN or room.room_type == Room.MINOR_BATHROOM or room.room_type == Room.MAIN_BATHROOM:
                duct_adjacent_to.append(
                    add_adjacency_constraint(duct, room, 0))
        model.Add(sum(duct_adjacent_to) > 0)


def add_floor_corridor_constraints(apartments, floor_corridors):
    '''There has to be at least one corridor for the floor'''
    assert(len(floor_corridors) > 0)

    n_floor_corridors = len(floor_corridors)
    # Corridor 0 is the main corridor and all other corridors are adjacent to it
    for i in range(1, n_floor_corridors):
        add_adjacency_constraint(floor_corridors[i], floor_corridors[0])

    # The main corridor for each apartment is adjacent to one of the floor corridors
    for apartment in apartments:
        adjacent_to_corridors = []
        for room in apartment:
            if room.room_type == Room.CORRIDOR:
                for corridor in floor_corridors:
                    adjacent_to_corridors.append(
                        add_adjacency_constraint(room, corridor, 0))
                break

        model.Add(sum(adjacent_to_corridors) > 0)


def add_stair_elevator_constraints(stair, elevator, floor_corridors):

    stair_adjacent_to = []
    elevator_adjacent_to = []

    for floor_corridor in floor_corridors:
        stair_adjacent_to.append(
            add_adjacency_constraint(stair, floor_corridor, 0))
        elevator_adjacent_to.append(
            add_adjacency_constraint(elevator, floor_corridor, 0))

    model.Add(sum(stair_adjacent_to) > 0)
    model.Add(sum(elevator_adjacent_to) > 0)


def get_apartment_main_corridor(apartment):
    for room in apartment:
        if room.room_type == Room.CORRIDOR:
            return room


def add_elevator_distance_constraint(elevator, apartments):
    distance_to_elevator = []

    for apartment in apartments:
        main_corridor = get_apartment_main_corridor(apartment)
        distance_to_elevator.append(main_corridor.distance(elevator))

    for i in range(len(distance_to_elevator) - 1):
        model.Add(distance_to_elevator[i] == distance_to_elevator[i + 1])


def max_distance_to_bathroom(apartment):
    main_bathroom = None

    for room in apartment:
        if room.room_type == Room.MAIN_BATHROOM:
            main_bathroom = room

    assert(main_bathroom)
    living_room_distance = None
    others_distance = []
    max_others_distance = model.NewIntVar(0, FLOOR_WIDTH + FLOOR_LENGTH, '')

    for room in apartment:
        if room.room_type == Room.LIVING_ROOM:
            living_room_distance = room.distance(main_bathroom)
        else:
            others_distance.append(room.distance(main_bathroom))

    model.AddMaxEquality(max_others_distance, others_distance)
    if living_room_distance:
        living_room_distance_x2 = model.NewIntVar(
            0, 2*(FLOOR_LENGTH + FLOOR_WIDTH), '')
        model.AddMultiplicationEquality(living_room_distance_x2, [
                                        2, living_room_distance])
        summation = model.NewIntVar(0, 3*(FLOOR_LENGTH + FLOOR_WIDTH), '')
        model.Add(summation == living_room_distance_x2+max_others_distance)
        return summation
    else:
        return max_others_distance
    # return living_room_distance, max_others_distance


def get_bedrooms(apartment):
    bedrooms = []

    for room in apartment:
        if room.room_type == Room.BEDROOM:
            bedrooms.append(room)

    return bedrooms


def add_bedrooms_constraint(apartment):
    bedrooms = get_bedrooms(apartment)

    bedroom_distances = [0]

    for i in range(len(bedrooms)):
        for j in range(i):
            bedroom_distances.append(bedrooms[i].distance(bedrooms[j]))

    max_bedrooms_distance = model.NewIntVar(0, FLOOR_LENGTH + FLOOR_WIDTH, '')
    model.AddMaxEquality(max_bedrooms_distance, bedroom_distances)

    return max_bedrooms_distance


def enforce_distance_constraint(first_room, second_room, distance, equality):
    assert(distance <= FLOOR_LENGTH + FLOOR_WIDTH)
    assert(equality == Equality.GREATER_THAN or equality == Equality.LESS_THAN)

    rooms_distance = model.NewIntVar(0, FLOOR_LENGTH + FLOOR_WIDTH, '')
    rooms_distance = first_room.distance(second_room)

    satisfied = model.NewBoolVar('')

    if equality == Equality.GREATER_THAN:
        model.Add(rooms_distance > distance).OnlyEnforceIf(satisfied)
        model.Add(rooms_distance <= distance).OnlyEnforceIf(satisfied.Not())
        return satisfied

    model.Add(rooms_distance < distance).OnlyEnforceIf(satisfied)
    model.Add(rooms_distance >= distance).OnlyEnforceIf(satisfied.Not())

    return satisfied

# Takes in the flattened version of the apartments, universal.
# Consider corridors. For now it takes in all corridors.


def get_grid(flattened_floor):
    n = len(flattened_floor)
    grid = [[model.NewIntVar(-1, n-1, '') for j in range(FLOOR_WIDTH)]
            for i in range(FLOOR_LENGTH)]
    for i in range(FLOOR_LENGTH):
        for j in range(FLOOR_WIDTH):
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


def reorder_flattened_floor(flattened_floor):
    new_flattened_floor = []
    for room in flattened_floor:
        if not room in IGNORE_REACHABILITY:
            new_flattened_floor.append(room)
    consider_cnt = len(new_flattened_floor)
    for room in flattened_floor:
        if room in IGNORE_REACHABILITY:
            new_flattened_floor.append(room)
    return new_flattened_floor, consider_cnt


def get_reachability(grid, consider_cnt, floor_sides=[FloorSide.LANDSCAPE, FloorSide.OPEN]):
    """A cell is reachable if it's reachable from any of the four directions. """
    reachable_from = [[[model.NewBoolVar('%i %i %i' % (i, j, k)) for k in range(
        len(DI))] for j in range(FLOOR_WIDTH)] for i in range(FLOOR_LENGTH)]
    reachable = [[model.NewBoolVar('') for j in range(FLOOR_WIDTH)]
                 for i in range(FLOOR_LENGTH)]
    top = 0
    bottom = 1
    left = 2
    right = 3
    is_reachable = 1 if FLOOR_TOP_SIDE in floor_sides else 0

    for j in range(FLOOR_WIDTH):
        model.Add(reachable_from[0][j][top] == is_reachable)

    is_reachable = 1 if FLOOR_BOTTOM_SIDE in floor_sides else 0
    for j in range(FLOOR_WIDTH):
        model.Add(reachable_from[FLOOR_LENGTH - 1][j][bottom] == is_reachable)

    is_reachable = 1 if FLOOR_LEFT_SIDE in floor_sides else 0
    for i in range(FLOOR_LENGTH):
        model.Add(reachable_from[i][0][left] == is_reachable)

    is_reachable = 1 if FLOOR_RIGHT_SIDE in floor_sides else 0
    for i in range(FLOOR_LENGTH):
        model.Add(reachable_from[i][FLOOR_WIDTH-1][right] == is_reachable)

    for i in range(FLOOR_LENGTH):
        for j in range(FLOOR_WIDTH):
            current_cell = []
            # if (reachable[neighbor] == 1 and (equal to neighbor or neighbor is empty or neighbor is ignored))
            for k in range(len(DI)):
                i2 = i+DI[k]
                j2 = j+DJ[k]
                if(i2 < 0 or i2 >= FLOOR_LENGTH):
                    current_cell.append(reachable_from[i][j][k])
                    continue
                if(j2 < 0 or j2 >= FLOOR_WIDTH):
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
                neighbor_ignored = model.NewBoolVar('')
                model.Add(grid[i2][j2] >= consider_cnt).OnlyEnforceIf(
                    neighbor_ignored)
                model.Add(grid[i2][j2] < consider_cnt).OnlyEnforceIf(
                    neighbor_ignored.Not())
                not_blocked = model.NewBoolVar('')
                model.AddImplication(neighbor_empty, not_blocked)
                model.AddImplication(equal, not_blocked)
                model.AddImplication(neighbor_ignored, not_blocked)
                model.Add(not_blocked == 0).OnlyEnforceIf(
                    [equal.Not(), neighbor_empty.Not(), neighbor_ignored.Not()])
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


def add_reachability(reachability, grid, flattened_floor, room_idx, add=1):
    """For each room, one of its cells must be reachable."""

    is_cells_reachable = []
    for i in range(FLOOR_LENGTH):
        for j in range(FLOOR_WIDTH):
            # if grid[i][j]==index and sun_reachability[i][j]==1 then true
            is_cell_reachable = model.NewBoolVar('')
            in_room = model.NewBoolVar('')
            model.Add(grid[i][j] == room_idx).OnlyEnforceIf(in_room)
            model.Add(grid[i][j] != room_idx).OnlyEnforceIf(in_room.Not())
            model.Add(is_cell_reachable == 1).OnlyEnforceIf(
                [in_room, reachability[i][j]])
            model.AddImplication(
                in_room.Not(), is_cell_reachable.Not())
            model.AddImplication(
                reachability[i][j].Not(), is_cell_reachable.Not())
            is_cells_reachable.append(is_cell_reachable)

    is_room_reachable = model.NewBoolVar('')
    model.Add(sum(is_cells_reachable) > 0).OnlyEnforceIf(is_room_reachable)
    model.Add(sum(is_cells_reachable) == 0).OnlyEnforceIf(
        is_room_reachable.Not())

    if add == 1:
        model.Add(is_room_reachable == 1)

    return is_room_reachable


def add_soft_sun_reachability_constraint(sun_reachability, grid, apartment):
    is_room_sun_reachable = []
    un_important_rooms = [Room.CORRIDOR, Room.DUCT, Room.ELEVATOR,
                          Room.STAIR, Room.MAIN_BATHROOM, Room.MINOR_BATHROOM]
    for room_idx, room in enumerate(apartment):
        if not room.room_type in un_important_rooms:
            is_room_sun_reachable.append(add_reachability(sun_reachability, grid,
                                                          apartment, room_idx, 0))

    return sum(is_room_sun_reachable)


def add_sunroom_constraint(sun_reachability, grid, flattened_floor):
    for room_idx, room in enumerate(flattened_floor):
        if room.room_type == Room.SUNROOM:
            add_reachability(sun_reachability, grid,
                             flattened_floor, room_idx)

# If given |apartments| only, it will flatten the apartments.
# If given |apartments| and |floor_corridors| it will flatten both together.


def flatten_floor(apartments, ducts, floor_corridors, stair, elevator):
    flattened_floor = []

    for apartment in apartments:
        for room in apartment:
            flattened_floor.append(room)

    for corridor in floor_corridors:
        flattened_floor.append(corridor)

    for duct in ducts:
        flattened_floor.append(duct)

    flattened_floor.append(stair)
    flattened_floor.append(elevator)

    return flattened_floor

############ Debugging Functions ##################################


def check_grid(flattened_floor, grid):
    """Checks that the creeated Int var grid is equal to the visualized grid."""
    visited = [[False for j in range(FLOOR_WIDTH)]
               for i in range(FLOOR_LENGTH)]
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
    for i in range(FLOOR_LENGTH):
        for j in range(FLOOR_WIDTH):
            assert(visited[i][j] or solver.Value(grid[i][j]) == -1)


def print_sun_reachability(sun_reachibility):
    print('Sun Reachability Matrix')
    for i in range(FLOOR_LENGTH):
        for j in range(FLOOR_WIDTH):
            print(solver.Value(sun_reachability[i][j]), end=' ')
        print()


def visualize_floor(flattened_floor, grid, solver):
    """Visualizes the floor using matplotlib"""
    visualized_output = [[0 for j in range(FLOOR_WIDTH)]
                         for i in range(FLOOR_LENGTH)]
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

            if room_type == Room.DUCT:
                heatmap_text = ROOM_TYPE_MAP[str(room_type)]

            visualized_output[i][j] = value
            ax.text(
                j, i, heatmap_text, ha='center', va='center')

    im = ax.imshow(visualized_output)
    fig.tight_layout()
    plt.show()


def add_floor_utilization(grid):
    unitilized_cells = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            b = model.NewBoolVar('')
            model.Add(grid[i][j] == -1).OnlyEnforceIf(b)
            model.Add(grid[i][j] != -1).OnlyEnforceIf(b.Not())
            unitilized_cells.append(b)

    model.Add(sum(unitilized_cells) == 0)


def add_landsacpe_reachability(grid, apartments, consider_cnt):
    un_important_rooms = [Room.CORRIDOR, Room.DUCT, Room.ELEVATOR,
                          Room.STAIR, Room.MAIN_BATHROOM, Room.MINOR_BATHROOM]
    landsacpe_reachability = get_reachability(
        grid, consider_cnt, [FloorSide.LANDSCAPE])
    for apartment in apartments:
        landscape_reachability = []
        for room_idx, room in enumerate(apartment):
            if not room.room_type in un_important_rooms:
                landscape_reachability.append(add_reachability(landsacpe_reachability, grid,
                                                               flattened_floor, room_idx, 0))
        model.Add(sum(landscape_reachability) > 0)


def get_int_var_per_point(r1, c1, r2, c2, mirror, vertical=True):
    if(vertical):
        # r1 = r2, |c2-mirror|=|c1-mirror|
        equal_rows = model.NewBoolVar('')
        symmetric_cols = model.NewBoolVar('')
        model.Add(r1 == r2).OnlyEnforceIf(equal_rows)
        model.Add(r1 != r2).OnlyEnforceIf(equal_rows.Not())
        c1_diff = model.NewIntVar(-FLOOR_WIDTH, FLOOR_WIDTH, '')
        c2_diff = model.NewIntVar(-FLOOR_WIDTH, FLOOR_WIDTH, '')
        c1_abs_diff = model.NewIntVar(0, FLOOR_WIDTH, '')
        c2_abs_diff = model.NewIntVar(0, FLOOR_WIDTH, '')
        model.Add(c1_diff == c1 - mirror)
        model.Add(c2_diff == c2 - mirror)
        model.AddAbsEquality(c1_abs_diff, c1_diff)
        model.AddAbsEquality(c2_abs_diff, c2_diff)
        model.Add(c1_abs_diff == c2_abs_diff).OnlyEnforceIf(symmetric_cols)
        model.Add(c1_abs_diff != c2_abs_diff).OnlyEnforceIf(
            symmetric_cols.Not())
        return equal_rows, symmetric_cols
    else:
        # c1 = c2, |r2-mirror|=|r1-mirror|
        equal_cols = model.NewBoolVar('')
        symmetric_rows = model.NewBoolVar('')
        model.Add(c1 == c2).OnlyEnforceIf(equal_cols)
        model.Add(c1 != c2).OnlyEnforceIf(equal_cols.Not())
        r1_diff = model.NewIntVar(-FLOOR_LENGTH, FLOOR_LENGTH, '')
        r2_diff = model.NewIntVar(-FLOOR_LENGTH, FLOOR_LENGTH, '')
        r1_abs_diff = model.NewIntVar(0, FLOOR_LENGTH, '')
        r2_abs_diff = model.NewIntVar(0, FLOOR_LENGTH, '')
        model.Add(r1_diff == r1 - mirror)
        model.Add(r2_diff == r2 - mirror)
        model.AddAbsEquality(r1_abs_diff, r1_diff)
        model.AddAbsEquality(r2_abs_diff, r2_diff)
        model.Add(r1_abs_diff == r2_abs_diff).OnlyEnforceIf(symmetric_rows)
        model.Add(r1_abs_diff != r2_abs_diff).OnlyEnforceIf(
            symmetric_rows.Not())
        return equal_cols, symmetric_rows


def get_symmetry_corners(room_1, room_2, mirror, vertical=True):
    if vertical:
        b1, b2 = get_int_var_per_point(
            room_1.start_row, room_1.start_col, room_2.start_row, room_2.end_col, mirror, vertical)
        b3, b4 = get_int_var_per_point(
            room_1.end_row, room_1.end_col, room_2.end_row, room_2.start_col, mirror, vertical)
        return b1, b2, b3, b4
    else:
        b1, b2 = get_int_var_per_point(
            room_1.start_row, room_1.start_col, room_2.end_row, room_2.start_col, mirror, vertical)
        b3, b4 = get_int_var_per_point(
            room_1.end_row, room_1.end_col, room_2.start_row, room_2.end_col, mirror, vertical)
        return b1, b2, b3, b4


def add_symmetry(apartment_1, apartment_2):
    vertical_symmetry_flags = []
    vertical_mirror = model.NewIntVar(0, FLOOR_WIDTH, '')  # x= mirror
    for i in range(len(apartment_1)):
        room_1 = apartment_1[i]
        room_2 = apartment_2[i]
        b1, b2, b3, b4 = get_symmetry_corners(room_1, room_2, vertical_mirror)
        vertical_symmetry_flags.append(b1)
        vertical_symmetry_flags.append(b2)
        vertical_symmetry_flags.append(b3)
        vertical_symmetry_flags.append(b4)
        left_of_mirror = model.NewBoolVar('')
        model.Add(room_1.end_col <= vertical_mirror).OnlyEnforceIf(
            left_of_mirror)
        model.Add(room_1.end_col > vertical_mirror).OnlyEnforceIf(
            left_of_mirror.Not())
        vertical_symmetry_flags.append(left_of_mirror)
        right_of_mirror = model.NewBoolVar('')
        model.Add(room_2.start_col >= vertical_mirror).OnlyEnforceIf(
            right_of_mirror)
        model.Add(room_2.start_col < vertical_mirror).OnlyEnforceIf(
            right_of_mirror.Not())
        vertical_symmetry_flags.append(right_of_mirror)
    ########## horizontal mirror ###########
    horizontal_symmetry_flags = []
    horizontal_mirror = model.NewIntVar(0, FLOOR_LENGTH, '')  # y= mirror
    for i in range(len(apartment_1)):
        room_1 = apartment_1[i]
        room_2 = apartment_2[i]
        b1, b2, b3, b4 = get_symmetry_corners(
            room_1, room_2, horizontal_mirror, vertical=False)
        horizontal_symmetry_flags.append(b1)
        horizontal_symmetry_flags.append(b2)
        horizontal_symmetry_flags.append(b3)
        horizontal_symmetry_flags.append(b4)
        top_of_mirror = model.NewBoolVar('')
        model.Add(room_1.end_row <= horizontal_mirror).OnlyEnforceIf(
            top_of_mirror)
        model.Add(room_1.end_row > horizontal_mirror).OnlyEnforceIf(
            top_of_mirror.Not())
        horizontal_symmetry_flags.append(top_of_mirror)
        bottom_of_mirror = model.NewBoolVar('')
        model.Add(room_2.start_row >= horizontal_mirror).OnlyEnforceIf(
            bottom_of_mirror)
        model.Add(room_2.start_row < horizontal_mirror).OnlyEnforceIf(
            bottom_of_mirror.Not())
        horizontal_symmetry_flags.append(bottom_of_mirror)
    horizontal_symmetry = model.NewBoolVar('')
    vertical_symmetry = model.NewBoolVar('')

    model.AddBoolAnd(vertical_symmetry_flags).OnlyEnforceIf(
        vertical_symmetry)
    model.AddBoolAnd(horizontal_symmetry_flags).OnlyEnforceIf(
        horizontal_symmetry)
    model.Add(horizontal_symmetry + vertical_symmetry == 1)

########################   Main Method Starts Here   ########################

########################   Process Future Input Here ########################


def next_int():
    # next line contains only one integer
    return int(input_file.readline())


input_file = open("input.txt", "r")
# for line in f:
#     print(line)

line = input_file.readline().rstrip('\n').split(' ')
FLOOR_LENGTH = int(line[0])
FLOOR_WIDTH = int(line[1])
MAX_DIM = max(FLOOR_LENGTH, FLOOR_WIDTH)

floor_side_dict = {'L': FloorSide.LANDSCAPE,
                   'O': FloorSide.OPEN, 'N': FloorSide.NONE}

room_dict = {'bedroom': Room.BEDROOM, 'kitchen': Room.KITCHEN, 'diningroom': Room.DININGROOM,
             'mainbathroom': Room.MAIN_BATHROOM, 'minorbathroom': Room.MINOR_BATHROOM, 'livingroom': Room.LIVING_ROOM, 'dressingroom': Room.DRESSING_ROOM, 'sunroom': Room.SUNROOM}


line = input_file.readline().rstrip('\n').split(' ')
FLOOR_TOP_SIDE = floor_side_dict[line[0]]
FLOOR_RIGHT_SIDE = floor_side_dict[line[1]]
FLOOR_BOTTOM_SIDE = floor_side_dict[line[2]]
FLOOR_LEFT_SIDE = floor_side_dict[line[3]]
n_apartment_types = next_int()
apartments = []
apartment_corridors = []
ducts = []
model = cp_model.CpModel()
apartment_no = 0
apartment = []
cnt_per_apartment_type = []
soft_constraints = []
for apartment_type in range(n_apartment_types):
    cnt_per_apartment_type.append(next_int())
    n_rooms = next_int()
    n_corridors = next_int()
    rooms_parameters = []
    for i in range(n_rooms):
        line = input_file.readline().rstrip('\n').split(' ')
        assert(len(line) > 3)
        room_type = room_dict[line[0].lower()]
        min_area = int(line[1])
        width = int(line[2])
        height = int(line[3])
        adjacent_to = -1
        if room_type == Room.MINOR_BATHROOM or room_type == Room.DRESSING_ROOM:
            assert(len(line) == 5)
            adjacent_to = int(line[4])
        rooms_parameters.append(
            [room_type, min_area, width, height, adjacent_to])

    for soft_constraint_idx in range(4):
        if soft_constraint_idx == 1:
            line = input_file.readline().rstrip('\n').split(' ')
            assert(len(line) % 4 == 0)
            distance_soft_constraint = []
            for i in range(0, len(line), 4):
                idx1 = int(line[0])
                idx2 = int(line[1])
                equality = Equality.GREATER_THAN if line[2] == '>' else Equality.LESS_THAN
                distance = int(line[3])
                distance_soft_constraint.append(
                    [idx1, idx2, equality, distance])
            soft_constraints.append(distance_soft_constraint)
        else:
            soft_constraints.append([next_int()])

    for apartment_copy in range(cnt_per_apartment_type[apartment_type]):
        clone_apartment = []
        for room_parameters in rooms_parameters:
            clone_apartment.append(Rectangle(room_parameters[0], room_parameters[1],
                                             room_parameters[2], room_parameters[3], room_parameters[4], apartment=apartment_no))
        for corridor_num in range(n_corridors):
            clone_apartment.append(
                Rectangle(Room.CORRIDOR, apartment=apartment_no))
        apartments.append(clone_apartment)
        apartment_no = apartment_no + 1
        apartment_corridors.append(n_corridors)

n_apartments = len(apartments)
n_ducts = max(n_apartments - 1, 1)
for duct_no in range(n_ducts):
    ducts.append(Rectangle(Room.DUCT))


n_floor_corridors = 2
floor_corridors = []
for i in range(n_floor_corridors):
    floor_corridors.append(Rectangle(Room.CORRIDOR))

stair = Rectangle(Room.STAIR)
elevator = Rectangle(Room.ELEVATOR)
# ########################   Hard Constraints ########################
flattened_floor = flatten_floor(
    apartments, ducts, floor_corridors, stair, elevator)
flattened_floor, consider_cnt = reorder_flattened_floor(flattened_floor)
for apartment in apartments:
    for room in apartment:
        room.add_room_constraints(apartment)
add_no_intersection_constraint(flattened_floor)
for apartment_no, apartment in enumerate(apartments):
    add_corridor_constraint(apartment_corridors[apartment_no], apartment)
add_floor_corridor_constraints(apartments, floor_corridors)
add_stair_elevator_constraints(stair, elevator, floor_corridors)
grid = get_grid(flattened_floor)
add_floor_utilization(grid)
add_duct_constraints(ducts, flattened_floor)
sun_reachability = get_reachability(grid, consider_cnt)
add_sunroom_constraint(sun_reachability, grid, flattened_floor)
# ########################   Hard Constraints ########################

# ########################   Soft Constraints ########################
apartment_idx = 0
sunreachability_constraint = []
distances_between_pairs = []
bedroom_distances = [0]
distances_to_main_bathroom = []
for apartment_type in range(n_apartment_types):
    for i in range(cnt_per_apartment_type[apartment_type]):
        apartment = apartments[apartment_idx + i]
        if soft_constraints[apartment_type * 4][0] == 1:
            sunreachability_constraint.append(add_soft_sun_reachability_constraint(
                sun_reachability, grid, apartment))
        for distance_constraint in soft_constraints[apartment_type * 4 + 1]:
            idx1 = distance_constraint[0]
            idx2 = distance_constraint[1]
            equality = distance_constraint[2]
            distance = distance_constraint[3]
            distances_between_pairs.append(enforce_distance_constraint(
                apartment[idx1], apartment[idx2], distance, equality))
        if soft_constraints[apartment_type * 4 + 2][0] == 1:
            bedroom_distances.append(add_bedrooms_constraint(apartment))
        if soft_constraints[apartment_type * 4 + 3][0] == 1:
            distances_to_main_bathroom.append(
                max_distance_to_bathroom(apartment))
    apartment_idx += cnt_per_apartment_type[apartment_type]

# max_bedroom_distances = model.NewIntVar(0, FLOOR_LENGTH + FLOOR_WIDTH, '')
# max_distance_to_main_bathroom = model.NewIntVar(
    # 0, 3 * (FLOOR_LENGTH + FLOOR_WIDTH), '')
# model.AddMaxEquality(max_bedroom_distances, bedroom_distances)
# model.AddMaxEquality(max_distance_to_main_bathroom, distances_to_main_bathroom)
# model.Maximize(-1 * max_distance_to_main_bathroom)

# ########################   Soft Constraints ########################

# ########################   Global Constraints ########################

global_landscape_view = next_int()
global_elevator_distance = next_int()
gloabal_symmetry_constraint = next_int()
if global_landscape_view == 1:
    add_landsacpe_reachability(grid, apartments, consider_cnt)
if global_elevator_distance == 1:
    add_elevator_distance_constraint(elevator, apartments)
if gloabal_symmetry_constraint == 1:
    apartment_no = 0
    for apartment_type in range(n_apartment_types):
        if cnt_per_apartment_type[apartment_type] == 2:
            add_symmetry(apartments[apartment_no],
                         apartments[apartment_no + 1])
        apartment_no += cnt_per_apartment_type[apartment_type]


# ########################   Global Constraints ########################
solver = cp_model.CpSolver()
solution_printer = VarArraySolutionPrinterWithLimit(flattened_floor, grid, 2)
status = solver.SearchForAllSolutions(model, solution_printer)
print(solver.StatusName())
print('time = ', solver.WallTime())

########################   Main Method Ends Here   ##########################

########################  Debuging ################

# check_grid(flattened_floor, grid)

# visualize_floor(flattened_floor, grid)
