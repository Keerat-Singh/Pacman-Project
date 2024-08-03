from Ghost import C_Ghost
import Constants
import HelperFunction

# Old
# class C_Pinky(C_Ghost):

#     def __init__(self, board, pacman):
#         super().__init__(board, pacman)

#         self.rect.x, self.rect.y = Constants.INITIAL_PINKY_POSITION[Constants.MAP_INDEX]
#         self.initialx, self.initialy = Constants.INITIAL_PINKY_POSITION[Constants.MAP_INDEX]

#         # timer to switch state
#         self.state_switch_timer = 17


# Need to test
class C_Pinky(C_Ghost):
    def __init__(self, board, pacman):
        super().__init__(board, pacman)
        # self.ghost_color = Constants.PINK
        self.rect.x, self.rect.y = Constants.INITIAL_PINKY_POSITION[Constants.MAP_INDEX]
        self.initialx, self.initialy = Constants.INITIAL_PINKY_POSITION[Constants.MAP_INDEX]

        # timer to switch state
        self.state_switch_timer = 17

    def goal_update(self):
        if self.state < 2:
            pacman_pos = HelperFunction.current_position(self.pacman)
            pacman_direction = self.pacman.direction
            if pacman_direction == 'UP':
                target_pos = (pacman_pos[0], pacman_pos[1] - 4)
            elif pacman_direction == 'DOWN':
                target_pos = (pacman_pos[0], pacman_pos[1] + 4)
            elif pacman_direction == 'LEFT':
                target_pos = (pacman_pos[0] - 4, pacman_pos[1])
            elif pacman_direction == 'RIGHT':
                target_pos = (pacman_pos[0] + 4, pacman_pos[1])
            return target_pos
        else:
            return (self.initialx, self.initialy)