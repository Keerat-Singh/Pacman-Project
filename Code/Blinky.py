from Ghost import C_Ghost
import Constants

class C_Blinky(C_Ghost):

    def __init__(self, board):
        super().__init__(board)

        # Initial/place for ghost to rest/spawn
        self.rect.center = ((Constants.GHOST_SIZE*20 - Constants.GHOST_SIZE*1.5), Constants.GHOST_SIZE*1.5)