from Ghost import C_Ghost
import Constants
import HelperFunction

# class C_Inky(C_Ghost):

#     def __init__(self, board, pacman):
#         super().__init__(board, pacman)

#         self.rect.x, self.rect.y = Constants.INITIAL_INKY_POSITION[Constants.MAP_INDEX]
#         self.initialx, self.initialy = Constants.INITIAL_INKY_POSITION[Constants.MAP_INDEX]

#         # timer to switch state
#         self.state_switch_timer = 6

# Need to test
class C_Inky(C_Ghost):
    def __init__(self, board, pacman, blinky):
        super().__init__(board, pacman)
        self.blinky = blinky
        # self.ghost_color = Constants.BLUE

        self.rect.x, self.rect.y = Constants.INITIAL_INKY_POSITION[Constants.MAP_INDEX]
        self.initialx, self.initialy = Constants.INITIAL_INKY_POSITION[Constants.MAP_INDEX]

        # timer to switch state
        self.state_switch_timer = 6

    def goal_update(self):
        if self.state < 2:
            pacman_pos = HelperFunction.current_position(self.pacman)
            pacman_direction = self.pacman.direction
            if pacman_direction == 'UP':
                target_pos = (pacman_pos[0], pacman_pos[1] - 2)
            elif pacman_direction == 'DOWN':
                target_pos = (pacman_pos[0], pacman_pos[1] + 2)
            elif pacman_direction == 'LEFT':
                target_pos = (pacman_pos[0] - 2, pacman_pos[1])
            elif pacman_direction == 'RIGHT':
                target_pos = (pacman_pos[0] + 2, pacman_pos[1])
            
            blinky_pos = HelperFunction.current_position(self.blinky)
            inky_target_pos = (2 * target_pos[0] - blinky_pos[0], 2 * target_pos[1] - blinky_pos[1])
            return inky_target_pos
        else:
            return (self.initialx, self.initialy)