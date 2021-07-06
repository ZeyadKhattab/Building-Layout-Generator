import constants
import globalVars
from enums import *
from project import add_adjacency_constraint, add_intersection_between_edges


class Rectangle:
    room_id = 1

    def __init__(self, room_type, min_area=1, width=0, height=0, adjacent_to=-1, apartment=-1):
        # Name the variable names in the model properly.
        FLOOR_LENGTH = globalVars.FLOOR_LENGTH
        FLOOR_WIDTH = globalVars.FLOOR_WIDTH
        model = globalVars.model
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
        if (width > 0):
            # print('width in ctor', width)
            # print('add_width')
            self.add_width(width)
        if (height > 0):
            self.add_height(height)
        model = globalVars.model
        model.Add(self.width == self.end_col - self.start_col)
        model.Add(self.height == self.end_row - self.start_row)
        model.AddMultiplicationEquality(self.area, [self.width, self.height])

    def add_room_constraints(self, apartment):
        adjacent_to = self.adjacent_to
        if self.room_type == Room.DININGROOM:
            for i in range(len(apartment)):
                if (apartment[i].room_type == Room.KITCHEN):
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
        globalVars.model.Add(self.width == width)

    def add_height(self, height):
        globalVars.model.Add(self.height == height)

    def get_left(self):
        return self.start_col

    def get_right(self):
        return self.end_col

    def get_top(self):
        return self.start_row

    def get_bottom(self):
        return self.end_row

    def distance(self, other):
        model = globalVars.model
        FLOOR_LENGTH = globalVars.FLOOR_LENGTH
        FLOOR_WIDTH = globalVars.FLOOR_WIDTH
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

        model.Add(dist == other.start_row - self.end_row + other.start_col -
                  self.end_col).OnlyEnforceIf([bottom, right])
        model.Add(dist == self.start_row - other.end_row + other.start_col -
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
