import pygame as PG
import random
import Constants
from GhostMovement import C_GhostMovement
import HelperFunction

# Ghost class
class C_Ghost(PG.sprite.Sprite):
    
    def __init__(self, board):
        super().__init__()
        self.board = board
        self.image = PG.Surface((Constants.GHOST_SIZE, Constants.GHOST_SIZE))
        self.image.fill(Constants.RED)
        self.rect = self.image.get_rect()

        # Initial/place for ghost to rest/spawn
        self.rect.center = ((Constants.PACMAN_SIZE*20 - Constants.PACMAN_SIZE*1.5), Constants.PACMAN_SIZE*1.5)
        # self.rect.x = random.randint(0, Constants.SCREEN_WIDTH - Constants.GHOST_SIZE)
        # self.rect.y = random.randint(0, Constants.SCREEN_HEIGHT - Constants.GHOST_SIZE)

        self.speed = Constants.GHOST_SPEED
        self.direction = random.choice(Constants.DIRECTION)
        self.state = 0
        self.frame_counter = 0
    
    # def update(self):
    #     C_GhostMovement.update(self= self)

    def update(self):
        # Update movement
        self.frame_counter += 1
        if self.frame_counter >= Constants.MOVE_DELAY:
            self.frame_counter = 0
            C_GhostMovement.update(self= self)

    def draw(self, screen):
        PG.draw.rect(screen, (255, 0, 0), self.rect)
    
    def can_move(self, ghost_x, ghost_y):
        return HelperFunction.can_move(self, ghost_x, ghost_y)