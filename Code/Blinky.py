from Ghost import C_Ghost
import Constants
import HelperFunction
import random

class C_Blinky(C_Ghost):
    def __init__(self, board, pacman):
        super().__init__(board, pacman)

        self.rect.x, self.rect.y = Constants.INITIAL_BLINKY_POSITION[Constants.MAP_INDEX]
        self.initialx, self.initialy = Constants.INITIAL_BLINKY_POSITION[Constants.MAP_INDEX]
        self.ghost_color = Constants.BLINKY_COLOR
        
        # looping path; we are caculating the smallest path when ghosts are generated; only once         
        self.smallest_loop_path = self.smallest_looping_path(self.initialx, self.initialy, board.map)
        print(f"Smallest loop path for Blinky: {self.smallest_loop_path}")

    def goal_update(self):
        if self.state < 2:
            return HelperFunction.current_position(self.pacman)
        else:
            return (self.initialx, self.initialy)  