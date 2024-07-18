import pygame as PG
import Constants
import random

class C_GhostMovement():
    
    # direction = random.choice(Constants.DIRECTION)

    def update(self):

        # Random movement; which will be updated for each ghost seperately
        if self.direction == 'LEFT':
            self.rect.x -= self.speed
        elif self.direction == 'RIGHT':
            self.rect.x += self.speed
        elif self.direction == 'UP':
            self.rect.y -= self.speed
        elif self.direction == 'DOWN':
            self.rect.y += self.speed

        # Randomly change direction
        if random.random() < 0.1:  # Change direction 2% of the time
            self.direction = random.choice(Constants.DIRECTION)

        # Keep Ghost within the screen boundaries
        if self.rect.left < 0:
            self.rect.left = 0
            self.direction = 'RIGHT'
        if self.rect.right > Constants.SCREEN_WIDTH:
            self.rect.right = Constants.SCREEN_WIDTH
            self.direction = 'LEFT'
        if self.rect.top < 0:
            self.rect.top = 0
            self.direction = 'DOWN'
        if self.rect.bottom > Constants.SCREEN_HEIGHT:
            self.rect.bottom = Constants.SCREEN_HEIGHT
            self.direction = 'UP'

            
        # Need to check for the Board border 