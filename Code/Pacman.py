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
        # self.image.fill(Constants.YELLOW)
        self.rect = self.image.get_rect()
        self.rect.center = (Constants.PACMAN_SIZE*1.5, Constants.PACMAN_SIZE*1.5)
        self.speed = Constants.PACMAN_SPEED

    def draw(self, screen):
        PG.draw.circle(screen, (255, 255, 0), self.rect.center, self.rect.width // 2)      

    def update(self):
        super().update()