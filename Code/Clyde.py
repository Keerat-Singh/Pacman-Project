from Ghost import C_Ghost
import Constants
import HelperFunction

# Old
# class C_Clyde(C_Ghost):

#     def __init__(self, board, pacman):
#         super().__init__(board, pacman)

#         self.rect.x, self.rect.y = Constants.INITIAL_CLYDE_POSITION[Constants.MAP_INDEX]
#         self.initialx, self.initialy = Constants.INITIAL_CLYDE_POSITION[Constants.MAP_INDEX]

#         # timer to switch state
#         self.state_switch_timer = 14


# Need to test
class C_Clyde(C_Ghost):
    def __init__(self, board, pacman):
        super().__init__(board, pacman)
        # self.ghost_color = Constants.ORANGE
        self.rect.x, self.rect.y = Constants.INITIAL_CLYDE_POSITION[Constants.MAP_INDEX]
        self.initialx, self.initialy = Constants.INITIAL_CLYDE_POSITION[Constants.MAP_INDEX]

        # timer to switch state
        self.state_switch_timer = 14

    def goal_update(self):
        if self.state < 2:
            pacman_pos = HelperFunction.current_position(self.pacman)
            ghost_pos = (self.rect.x, self.rect.y)
            if HelperFunction.calculate_distance(ghost_pos, pacman_pos) > 8:
                return pacman_pos
            else:
                return (self.initialx, self.initialy)
        else:
            return (self.initialx, self.initialy)