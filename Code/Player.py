import pygame as PG
import Constants
import random
from Controller import C_Controller
import HelperFunction

# Player class
class C_Player(PG.sprite.Sprite):

    def __init__(self):
        super().__init__()
        self.frame_counter = 0


    def update(self):
        self.frame_counter += 1
        if self.frame_counter >= Constants.MOVE_DELAY:
            self.frame_counter = 0
            C_Controller.update(self= self)
    
    def can_move(self, new_x, new_y):

        # # Check if the new position is within the screen boundaries
        # if new_x < 0 or new_x >= len(self.board.map[0]) or new_y < 0 or new_y >= len(self.board.map):
        #     return False
        # # Check if the new position is a valid cell (1)
        # return self.board.map[new_y][new_x] == 0

        return HelperFunction.can_move(self, new_x, new_y)
    
    

