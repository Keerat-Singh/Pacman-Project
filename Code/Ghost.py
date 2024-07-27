import pygame as PG
import random
import Constants
from GhostMovement import C_GhostMovement
import HelperFunction

# Ghost class
class C_Ghost(PG.sprite.Sprite):
    
    def __init__(self, board, pacman):
        super().__init__()
        self.board = board
        self.image = PG.Surface((Constants.GHOST_SIZE, Constants.GHOST_SIZE))
        self.rect = self.image.get_rect()
        self.pacman = pacman

        # Initial/place for ghost to rest/spawn
        # Need to give some values
        self.rect.x = 20
        self.rect.y = 1

        self.direction = random.choice(Constants.DIRECTION)
        # self.state = Constants.GHOST_STATE[0]
        self.state = "NOT_CHASE"
        self.being_chased_timer = 20
        self.death_timer = 10
        self.ghost_color = ()

        # For movement
        self.frame_counter = 0

        # Astar components
        self.astar_helper = HelperFunction.AStar(board.map)
        self.path = []

    def update(self):
        # Update movement
        self.frame_counter += 1
        if self.frame_counter >= Constants.MOVE_DELAY + Constants.SPEED_DIFFERENCE:
            self.frame_counter = 0

            # pacman_pos = HelperFunction.current_position(self.pacman)
            # ghost_pos = (self.rect.x, self.rect.y)
            # self.path = self.astar_helper.find_path(ghost_pos, pacman_pos)
            # if self.path:
            #     next_pos = self.path.pop(0)
            #     # if self.can_move(next_pos[0], next_pos[1]):
            #     if HelperFunction.can_move(self, next_pos[0], next_pos[1]):
            #         self.rect.x = next_pos[0]
            #         self.rect.y = next_pos[1]

            # Movement
            C_GhostMovement.update(self= self)

    def draw(self, screen):
        # Ghost is not dead
        if self.state != Constants.GHOST_STATE[3]:
            # Ghost is not being chased
            if self.state != Constants.GHOST_STATE[2]:
                self.ghost_color = Constants.RED
            else:
                self.ghost_color = Constants.GREY
            # PG.draw.rect(screen, self.ghost_color, self.rect)
            PG.draw.rect(screen, self.ghost_color, PG.Rect(self.rect.x * Constants.BOARD_SIZE, 
                                                           self.rect.y * Constants.BOARD_SIZE, 
                                                           Constants.BOARD_SIZE, Constants.BOARD_SIZE))

    def dead_position(self):
        self.rect.x = -Constants.GHOST_SIZE
        self.rect.y = -Constants.GHOST_SIZE
        
    # Updating ghost state 
    def update_state(self, index):
        self.state = Constants.GHOST_STATE[index]
        if index == 3:
            self.dead_position()

    def current_state(self):
        return self.state.strip()