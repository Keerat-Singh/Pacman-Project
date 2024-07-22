import pygame as PG
import Constants
import random

class C_GhostMovement():

    def update(self):

        # Calculate the potential new position based on input
        ghost_x, ghost_y = self.rect.x // Constants.SIZE, self.rect.y // Constants.SIZE
        if self.direction == 'LEFT':
            ghost_x -= 1 
        elif self.direction == 'RIGHT':
            ghost_x += 1
        elif self.direction == 'UP':
            ghost_y -= 1
        elif self.direction == 'DOWN':
            ghost_y += 1

        # Randomly change direction
        if random.random() < 0.1:  # Change direction
            self.direction = random.choice(Constants.DIRECTION)

        # Check if the new position is valid
        if self.can_move(ghost_x, ghost_y):
            self.rect.x = ghost_x * Constants.SIZE
            self.rect.y = ghost_y * Constants.SIZE