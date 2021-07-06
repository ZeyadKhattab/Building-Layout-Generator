from Rectangle import Rectangle
from enums import Room
import globalVars
from IO import takeInput
import project
from project import add_soft_sun_reachability_constraint, flatten_floor, reorder_flattened_floor, \
    add_no_intersection_constraint, add_corridor_constraint, add_floor_corridor_constraints, \
    add_stair_elevator_constraints, get_grid, add_floor_utilization, add_duct_constraints, get_reachability, \
    add_sunroom_constraint, add_bedrooms_constraint, max_distance_to_bathroom, enforce_distance_constraint, \
    visualize_floor, add_symmetry, add_elevator_distance_constraint, add_landsacpe_reachability, add_divine_ratio

# ########################   Hard Constraints ########################
globalVars.init()
takeInput()
project.init()
globalVars.stair = Rectangle(Room.STAIR)
globalVars.elevator = Rectangle(Room.ELEVATOR)
flattened_floor = globalVars.flattened_floor
flattened_floor = flatten_floor(
    globalVars.apartments, globalVars.ducts, globalVars.floor_corridors, globalVars.stair, globalVars.elevator)
flattened_floor, consider_cnt = reorder_flattened_floor(flattened_floor)
for apartment in globalVars.apartments:
    for room in apartment:
        room.add_room_constraints(apartment)
add_no_intersection_constraint(flattened_floor)
for apartment_no, apartment in enumerate(globalVars.apartments):
    add_corridor_constraint(globalVars.apartment_corridors[apartment_no], apartment)
add_floor_corridor_constraints(globalVars.apartments, globalVars.floor_corridors)
add_stair_elevator_constraints(globalVars.stair, globalVars.elevator, globalVars.floor_corridors)
grid = get_grid(flattened_floor)
add_floor_utilization(grid)
add_duct_constraints(globalVars.ducts, flattened_floor)
sun_reachability = get_reachability(grid, consider_cnt)
add_sunroom_constraint(sun_reachability, grid, flattened_floor)
# ########################   Hard Constraints ########################

# ########################   Soft Constraints ########################
apartment_idx = 0
sunreachability_constraint = []
distances_between_pairs = []
bedroom_distances = [0]
distances_to_main_bathroom = [0]
for apartment_type in range(globalVars.n_apartment_types):
    for i in range(globalVars.cnt_per_apartment_type[apartment_type]):
        apartment = globalVars.apartments[apartment_idx + i]
        if globalVars.soft_constraints[apartment_type * 4][0] == 1:
            sunreachability_constraint.append(add_soft_sun_reachability_constraint(
                sun_reachability, grid, apartment))
        for distance_constraint in globalVars.soft_constraints[apartment_type * 4 + 1]:
            if len(distance_constraint) == 0:
                continue
            idx1 = distance_constraint[0]
            idx2 = distance_constraint[1]
            equality = distance_constraint[2]
            distance = distance_constraint[3]
            distances_between_pairs.append(enforce_distance_constraint(
                apartment[idx1], apartment[idx2], distance, equality))
        if globalVars.soft_constraints[apartment_type * 4 + 2][0] == 1:
            bedroom_distances.append(add_bedrooms_constraint(apartment))
        if globalVars.soft_constraints[apartment_type * 4 + 3][0] == 1:
            distances_to_main_bathroom.append(
                max_distance_to_bathroom(apartment))
    apartment_idx += globalVars.cnt_per_apartment_type[apartment_type]

max_bedroom_distances = globalVars.model.NewIntVar(0, globalVars.FLOOR_LENGTH + globalVars.FLOOR_WIDTH, '')
max_distance_to_main_bathroom = globalVars.model.NewIntVar(
    0, 3 * (globalVars.FLOOR_LENGTH + globalVars.FLOOR_WIDTH), '')
globalVars.model.AddMaxEquality(max_bedroom_distances, bedroom_distances)
globalVars.model.AddMaxEquality(max_distance_to_main_bathroom, distances_to_main_bathroom)
globalVars.model.Maximize(sum(distances_between_pairs) + -1 * max_bedroom_distances + -
1 * max_distance_to_main_bathroom + sum(sunreachability_constraint))

# ########################   Soft Constraints ########################

# ########################   Global Constraints ########################

global_landscape_view = globalVars.global_landscape_view
global_elevator_distance = globalVars.global_elevator_distance
gloabal_symmetry_constraint = globalVars.gloabal_symmetry_constraint
global_divine_ratio = globalVars.global_divine_ratio
if global_landscape_view == 1:
    add_landsacpe_reachability(grid, globalVars.apartments, consider_cnt)
if global_elevator_distance == 1:
    add_elevator_distance_constraint(globalVars.elevator, globalVars.apartments)
if gloabal_symmetry_constraint == 1:
    apartment_no = 0
    for apartment_type in range(globalVars.n_apartment_types):
        if globalVars.cnt_per_apartment_type[apartment_type] == 2:
            add_symmetry(globalVars.apartments[apartment_no],
                         globalVars.apartments[apartment_no + 1])
        apartment_no += globalVars.cnt_per_apartment_type[apartment_type]
if global_divine_ratio == 1:
    add_divine_ratio(flattened_floor)

# ########################   Global Constraints ########################

solver = globalVars.cp_model.CpSolver()
# solution_printer = VarArraySolutionPrinterWithLimit(flattened_floor, grid, 2)
# status = solver.SearchForAllSolutions(model, solution_printer)
solver.Solve(globalVars.model)

print(solver.StatusName())
print('time = ', solver.WallTime())

apartment_idx = 0
for apartment_type in range(globalVars.n_apartment_types):
    for i in range(globalVars.cnt_per_apartment_type[apartment_type]):
        print('Soft constraints stats for apartment %d in apartment type %d:' % (
            (i + 1), (apartment_type + 1)))
        apartment = globalVars.apartments[apartment_idx + i]
        if globalVars.soft_constraints[apartment_type * 4][0] == 1:
            print('Number of rooms reachable from the sun: ',
                  solver.Value(sunreachability_constraint[apartment_idx + i]))
        for distance_constraint in globalVars.soft_constraints[apartment_type * 4 + 1]:
            if len(distance_constraint) == 0:
                continue
            print('Number of distane constraints satisfied: ', solver.Value(
                distances_between_pairs[apartment_idx + i]))
        if globalVars.soft_constraints[apartment_type * 4 + 2][0] == 1:
            print('Max distance between bedrooms: ', solver.Value(
                bedroom_distances[apartment_idx + i]))
        if globalVars.soft_constraints[apartment_type * 4 + 3][0] == 1:
            print('Max distance to the main bathroom: ',
                  solver.Value(distances_to_main_bathroom[apartment_idx + i]))
    apartment_idx += globalVars.cnt_per_apartment_type[apartment_type]

visualize_floor(flattened_floor, grid, solver)

########################   Main Method Ends Here   ########################
