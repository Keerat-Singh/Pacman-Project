import pygame as PG
import random
import Constants
# from GhostMovement import C_GhostMovement

# Ghost class
class C_Ghost(PG.sprite.Sprite):
    
    def __init__(self):
        super().__init__()
        self.image = PG.Surface((Constants.GHOST_SIZE, Constants.GHOST_SIZE))
        self.image.fill(Constants.RED)
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, Constants.SCREEN_WIDTH - Constants.GHOST_SIZE)
        self.rect.y = random.randint(0, Constants.SCREEN_HEIGHT - Constants.GHOST_SIZE)
        self.speed = Constants.GHOST_SPEED
        self.direction = random.choice(Constants.DIRECTION)
    
    # def update(self):
    #     C_GhostMovement.update(self= self)

    def update(self):

        print(f"Before movement: {self.rect}, direction: {self.direction}")

        # Random movement; which will be updated for each ghost separately
        if self.direction == 'LEFT':
            self.rect.x -= self.speed
        elif self.direction == 'RIGHT':
            self.rect.x += self.speed
        elif self.direction == 'UP':
            self.rect.y -= self.speed
        elif self.direction == 'DOWN':
            self.rect.y += self.speed

        # Randomly change direction
        if random.random() < 0.1:  # Change direction 10% of the time
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

        print(f"After movement: {self.rect}, direction: {self.direction}")