import pygame as PG
import Constants
import random
from Controller import C_Controller

# Player class
class C_Player(PG.sprite.Sprite):

    def __init__(self):
        super().__init__()
        self.image = PG.Surface((Constants.DEFAULT_SIZE, Constants.DEFAULT_SIZE))
        self.image.fill(Constants.YELLOW)
        self.rect = self.image.get_rect()
        self.rect.center = (Constants.SCREEN_WIDTH // 2, Constants.SCREEN_HEIGHT // 2)              
        self.speed = Constants.DEFAULT_SPEED

    def update(self):
        C_Controller.update(self= self)