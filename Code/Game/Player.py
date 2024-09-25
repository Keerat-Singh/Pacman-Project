import pygame as PG
from . import Constants
from .Controller import C_Controller

# Player class
class C_Player(PG.sprite.Sprite):

    def __init__(self):
        super().__init__()
        self.frame_counter = 0


    def update(self):
        self.frame_counter += 1
        if self.frame_counter >= Constants.MOVE_DELAY:
            self.frame_counter = 0
            C_Controller.update(self)