import pygame as PG
import Constants
import random

class C_Controller():
    
    # def update(self):

    #     # Control
    #     keys = PG.key.get_pressed()
    #     if keys[PG.K_LEFT]:
    #         self.rect.x -= self.speed
    #     elif keys[PG.K_RIGHT]:
    #         self.rect.x += self.speed
    #     elif keys[PG.K_UP]:
    #         self.rect.y -= self.speed
    #     elif keys[PG.K_DOWN]:
    #         self.rect.y += self.speed

    #     # Keep Pac-Man within the screen boundaries
    #     if self.rect.left < 0:
    #         self.rect.left = 0
    #     if self.rect.right > Constants.SCREEN_WIDTH:
    #         self.rect.right = Constants.SCREEN_WIDTH
    #     if self.rect.top < 0:
    #         self.rect.top = 0
    #     if self.rect.bottom > Constants.SCREEN_HEIGHT:
    #         self.rect.bottom = Constants.SCREEN_HEIGHT


    def update(self):
        # Calculate the potential new position based on input
        new_x, new_y = self.rect.x // Constants.SIZE, self.rect.y // Constants.SIZE
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
        if self.can_move(new_x, new_y):
            self.rect.x = new_x * Constants.SIZE
            self.rect.y = new_y * Constants.SIZE
    