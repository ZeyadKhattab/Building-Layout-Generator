from enum import Enum


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
