from Ghost import C_Ghost
import Constants

class C_Clyde(C_Ghost):

    def __init__(self, board, pacman):
        super().__init__(board, pacman)

        self.rect.x, self.rect.y = Constants.INITIAL_CLYDE_POSITION[Constants.MAP_INDEX]
        self.initialx, self.initialy = Constants.INITIAL_CLYDE_POSITION[Constants.MAP_INDEX]

        # timer to switch state
        self.state_switch_timer = 14