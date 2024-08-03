from Ghost import C_Ghost
import Constants
import HelperFunction
import random

# Old
# class C_Blinky(C_Ghost):

#     def __init__(self, board, pacman):
#         super().__init__(board, pacman)

#         self.rect.x, self.rect.y = Constants.INITIAL_BLINKY_POSITION[Constants.MAP_INDEX]
#         self.initialx, self.initialy = Constants.INITIAL_BLINKY_POSITION[Constants.MAP_INDEX]

#         # timer to switch state
#         self.state_switch_timer = 10


# Need to test
class C_Blinky(C_Ghost):
    def __init__(self, board, pacman):
        super().__init__(board, pacman)
        # self.ghost_color = Constants.RED
        self.rect.x, self.rect.y = Constants.INITIAL_BLINKY_POSITION[Constants.MAP_INDEX]
        self.initialx, self.initialy = Constants.INITIAL_BLINKY_POSITION[Constants.MAP_INDEX]

        # timer to switch state
        self.state_switch_timer = 10

    def goal_update(self):
        if self.state < 2:
            return HelperFunction.current_position(self.pacman)
        else:
            return (self.initialx, self.initialy)