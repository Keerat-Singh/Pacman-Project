import pygame as PG
from . import HelperFunction

class C_Controller():

    def update(self):
        # Calculate the potential new position based on input
        new_x, new_y = self.rect.x, self.rect.y
        keys = PG.key.get_pressed()
        if keys[PG.K_LEFT]:
            new_x -= 1
            self.direction = 'LEFT'
        elif keys[PG.K_RIGHT]:
            new_x += 1
            self.direction = 'RIGHT'
        elif keys[PG.K_UP]:
            new_y -= 1
            self.direction = 'UP'
        elif keys[PG.K_DOWN]:
            new_y += 1
            self.direction = 'DOWN'

        # Check if the new position is valid
        if HelperFunction.can_move(self, new_x, new_y):
            self.rect.x = new_x
            self.rect.y = new_y