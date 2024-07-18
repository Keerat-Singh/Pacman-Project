import pygame as PG
import Constants
import random

class C_Controller():
    
    def update(self):
        keys = PG.key.get_pressed()
        if keys[PG.K_LEFT]:
            self.rect.x -= self.speed
        if keys[PG.K_RIGHT]:
            self.rect.x += self.speed
        if keys[PG.K_UP]:
            self.rect.y -= self.speed
        if keys[PG.K_DOWN]:
            self.rect.y += self.speed

        # Keep Pac-Man within the screen boundaries
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > Constants.SCREEN_WIDTH:
            self.rect.right = Constants.SCREEN_WIDTH
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > Constants.SCREEN_HEIGHT:
            self.rect.bottom = Constants.SCREEN_HEIGHT