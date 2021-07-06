from enums import *


floor_side_dict = {'L': FloorSide.LANDSCAPE,
                   'O': FloorSide.OPEN, 'N': FloorSide.NONE}

room_dict = {'bedroom': Room.BEDROOM, 'kitchen': Room.KITCHEN, 'diningroom': Room.DININGROOM,
             'mainbathroom': Room.MAIN_BATHROOM, 'minorbathroom': Room.MINOR_BATHROOM, 'livingroom': Room.LIVING_ROOM,
             'dressingroom': Room.DRESSING_ROOM, 'sunroom': Room.SUNROOM}

DI = [-1, 1, 0, 0]
DJ = [0, 0, -1, 1]

IGNORE_REACHABILITY = [Room.DUCT]
ROOM_TYPE_MAP = {'Room.DININGROOM': 'DR', 'Room.KITCHEN': 'KT', 'Room.MAIN_BATHROOM': 'MB', 'Room.MINOR_BATHROOM': 'mb',
                 'Room.DRESSING_ROOM': 'DRS', 'Room.BEDROOM': 'BD', 'Room.SUNROOM': 'SR', 'Room.CORRIDOR': 'C',
                 'Room.DUCT': 'D', 'Room.STAIR': 'S', 'Room.ELEVATOR': 'E', 'Room.LIVING_ROOM': 'LR', 'Room.OTHER': 'X'}

UNIMPORTANT_ROOMS = [Room.CORRIDOR, Room.DUCT, Room.ELEVATOR,
                     Room.STAIR, Room.MAIN_BATHROOM, Room.MINOR_BATHROOM]



