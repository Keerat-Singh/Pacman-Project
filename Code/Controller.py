import pygame as PG
import Constants
import random
import HelperFunction

class C_Controller():

    def update(self):
        # Calculate the potential new position based on input
        new_x, new_y = self.rect.x, self.rect.y
        keys = PG.key.get_pressed()
        if keys[PG.K_LEFT]:
            new_x -= 1
        elif keys[PG.K_RIGHT]:
            new_x += 1
        elif keys[PG.K_UP]:
            new_y -= 1
        elif keys[PG.K_DOWN]:
            new_y += 1

        # Check if the new position is valid
        if HelperFunction.can_move(self, new_x, new_y):
            self.rect.x = new_x
            self.rect.y = new_y