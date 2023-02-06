import pygame
import numpy as np

class C_Tile:

# Initialising variables 
    wall = False
    dot = False
    bigDot = False
    eaten = False
    pos = np.array([0,0])

# Constructor
    def __init__(self):
        pass

#   Tile(float x, float y) {
#     pos = new PVector(x, y);
#   }

# Drawing a dot if there is a '1' in the tile

    def show(self):
        print(self.dot)
        if self.dot:
            if not self.eaten:
                pygame.draw.circle(color= (255,255,0))

#   void show() {
#     if (dot) {
#       if (!eaten) {//draw dot
#         fill(255, 255, 0);
#         noStroke();
#         ellipse(pos.x, pos.y, 3, 3);
#       }
#     } else if (bigDot) {
#       if (!eaten) {//draw big dot
#         fill(255, 255, 0);
#         noStroke();
#         ellipse(pos.x, pos.y, 6, 6);
#       }
#     }
#   }
