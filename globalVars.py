from ortools.sat.python import cp_model


def init():
    global FLOOR_LENGTH
    global FLOOR_WIDTH
    global MAX_DIM
    global apartments
    apartments = []
    global ducts
    ducts = []
    global floor_corridors
    floor_corridors = []
    global stair

    global elevator
    global apartment_corridors

    global n_apartment_types

    global cnt_per_apartment_type
    cnt_per_apartment_type = []
    global soft_constraints
    soft_constraints = []
    global apartment_corridors
    apartment_corridors = []
    global model
    model = cp_model.CpModel()
    global FLOOR_TOP_SIDE
    global FLOOR_RIGHT_SIDE
    global FLOOR_LEFT_SIDE
    global FLOOR_BOTTOM_SIDE
    global global_landscape_view
    global global_elevator_distance
    global gloabal_symmetry_constraint
    global global_divine_ratio
    global flattened_floor
    flattened_floor = []
