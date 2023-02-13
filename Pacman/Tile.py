import pygame
import numpy as np

class C_Tile:

# Initialising variables 
    # wall = False
    # dot = False
    # bigDot = False
    # eaten = False
    # pos = np.array([0,0])

    def __init__(self):
        self.wall = False
        self.dot = False
        self.bigDot = False
        self.eaten = False
        self.pos = np.array([0,0])

# Function to set pos value
    def setPos(self, x, y):
        self.pos = np.array([x, y])

# Drawing a dot if there is a '1' in the tile
    def show(self, SCREEN):
        if self.dot:
            if not self.eaten:
                pygame.draw.circle(SCREEN, (255,255,0), (self.pos[1] +8, self.pos[0] +8), 2)
        elif self.bigDot:
            if not self.eaten:
                pygame.draw.circle(SCREEN, (255,0,0), (self.pos[1] +8, self.pos[0] +8), 4)
        # elif self.wall:
        #     pygame.draw.rect(SCREEN, (0,0,255), pygame.Rect(self.pos[1],self.pos[0],16,16))
