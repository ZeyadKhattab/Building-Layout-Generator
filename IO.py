from Rectangle import Rectangle
from constants import *
from enums import *
import globalVars


def next_int(input_file):
    # next line contains only one integer
    return int(input_file.readline())


def next_line(input_file):
    return input_file.readline().rstrip('\n').split(' ')


def takeInput():
    input_file = open("input.txt", "r")
    line = next_line(input_file)

    globalVars.FLOOR_LENGTH = int(line[0])
    globalVars.FLOOR_WIDTH = int(line[1])
    globalVars.MAX_DIM = max(globalVars.FLOOR_LENGTH, globalVars.FLOOR_WIDTH)
    line = next_line(input_file)
    globalVars.FLOOR_TOP_SIDE = floor_side_dict[line[0]]
    globalVars.FLOOR_RIGHT_SIDE = floor_side_dict[line[1]]
    globalVars.FLOOR_BOTTOM_SIDE = floor_side_dict[line[2]]
    globalVars.FLOOR_LEFT_SIDE = floor_side_dict[line[3]]
    globalVars.n_apartment_types = next_int(input_file)

    apartment_no = 0

    for apartment_type in range(globalVars.n_apartment_types):
        globalVars.cnt_per_apartment_type.append(next_int(input_file))
        n_rooms = next_int(input_file)
        n_corridors = next_int(input_file)
        rooms_parameters = []
        for i in range(n_rooms):
            line = next_line(input_file)
            assert (len(line) > 3)
            room_type = room_dict[line[0].lower()]
            min_area = int(line[1])
            width = int(line[2])
            height = int(line[3])
            adjacent_to = -1
            if room_type == Room.MINOR_BATHROOM or room_type == Room.DRESSING_ROOM:
                assert (len(line) == 5)
                adjacent_to = int(line[4])
            rooms_parameters.append(
                [room_type, min_area, width, height, adjacent_to])

        for soft_constraint_idx in range(4):
            if soft_constraint_idx == 1:
                line = next_line(input_file)
                distance_soft_constraint = []
                if len(line) == 1:
                    globalVars.soft_constraints.append(distance_soft_constraint)
                    continue
                assert (len(line) % 4 == 0)
                for i in range(0, len(line), 4):
                    idx1 = int(line[0])
                    idx2 = int(line[1])
                    equality = Equality.GREATER_THAN if line[2] == '>' else Equality.LESS_THAN
                    distance = int(line[3])
                    distance_soft_constraint.append(
                        [idx1, idx2, equality, distance])
                globalVars.soft_constraints.append(distance_soft_constraint)
            else:
                globalVars.soft_constraints.append([next_int(input_file)])

        for apartment_copy in range(globalVars.cnt_per_apartment_type[apartment_type]):
            clone_apartment = []
            for room_parameters in rooms_parameters:
                clone_apartment.append(Rectangle(room_parameters[0], room_parameters[1],
                                                 room_parameters[2], room_parameters[3], room_parameters[4],
                                                 apartment=apartment_no))
            for corridor_num in range(n_corridors):
                clone_apartment.append(
                    Rectangle(Room.CORRIDOR, apartment=apartment_no))
            globalVars.apartments.append(clone_apartment)
            apartment_no = apartment_no + 1
            globalVars.apartment_corridors.append(n_corridors)

    n_apartments = len(globalVars.apartments)
    n_ducts = max(n_apartments - 1, 1)
    for duct_no in range(n_ducts):
        globalVars.ducts.append(Rectangle(Room.DUCT))

    n_floor_corridors = 2
    for i in range(n_floor_corridors):
        globalVars.floor_corridors.append(Rectangle(Room.CORRIDOR))


    globalVars.global_landscape_view = next_int(input_file)
    globalVars.global_elevator_distance = next_int(input_file)
    globalVars.gloabal_symmetry_constraint = next_int(input_file)
    globalVars.global_divine_ratio = next_int(input_file)
