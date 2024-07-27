import pygame as PG
import random
import Constants
from Player import C_Player

# Pac-Man class
class C_Pacman(C_Player):

    def __init__(self, board):

        super().__init__()
        self.board = board
        self.image = PG.Surface((Constants.PACMAN_SIZE, Constants.PACMAN_SIZE))
        self.rect = self.image.get_rect()
        self.rect.x = Constants.INITIAL_PACMAN_POSITION[Constants.MAP_INDEX][0]
        self.rect.y = Constants.INITIAL_PACMAN_POSITION[Constants.MAP_INDEX][1]

    def draw(self, screen):  
        PG.draw.circle(screen, Constants.YELLOW, (self.rect.x * Constants.PACMAN_SIZE + Constants.PACMAN_SIZE//2, 
                                                  self.rect.y * Constants.PACMAN_SIZE + Constants.PACMAN_SIZE//2),
                                                  self.rect.width // 2)      

    def update(self):
        super().update()