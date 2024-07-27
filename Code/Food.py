import pygame as PG
import numpy as np
import Constants

class C_Food():

# Initialising variables 
    def __init__(self, x, y, cell):
        # self.eaten = False
        self.food_info = cell
        self.pos = np.array([x ,y])

# Drawing a dot if there is a '1' in the tile
    def show(self, screen):
        if self.food_info == 0:
            # Show normal food
            PG.draw.circle(screen, Constants.YELLOW, (self.pos[0] * Constants.BOARD_SIZE + Constants.SIZE//2, 
                                                        self.pos[1] * Constants.BOARD_SIZE + Constants.SIZE//2), 
                                                        Constants.BOARD_SIZE // 4)
        else:
            # Show power up
            PG.draw.circle(screen, Constants.GREEN, (self.pos[0] * Constants.BOARD_SIZE + Constants.SIZE//2, 
                                                        self.pos[1] * Constants.BOARD_SIZE + Constants.SIZE//2), 
                                                        Constants.BOARD_SIZE // 3)
                
    def destroy():
        pass