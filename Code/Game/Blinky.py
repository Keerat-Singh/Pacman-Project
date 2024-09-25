from . import Constants
from . import HelperFunction
from .Ghost import C_Ghost

class C_Blinky(C_Ghost):
    def __init__(self, board, pacman):
        super().__init__(board, pacman)

        self.rect.x, self.rect.y = Constants.INITIAL_BLINKY_POSITION[Constants.MAP_INDEX]
        self.initialx, self.initialy = Constants.INITIAL_BLINKY_POSITION[Constants.MAP_INDEX]
        self.ghost_color = Constants.BLINKY_COLOR
        
        # looping path; we are caculating the smallest path when ghosts are generated; only once         
        self.smallest_loop_path = self.finding_smallest_looping_path(self.initialx, self.initialy, board.map)
        self.current_path_index = 0

    def goal_update(self):
        if self.state < 2:
            return HelperFunction.current_position(self.pacman)
        else:
            return (self.initialx, self.initialy)  