import pygame as PG
import random
import Constants
from GhostMovement import C_GhostMovement
import HelperFunction
import time
import threading

# Ghost class
class C_Ghost(PG.sprite.Sprite):
    
    def __init__(self, board, pacman):
        super().__init__()
        self.board = board
        self.image = PG.Surface((Constants.GHOST_SIZE, Constants.GHOST_SIZE))
        self.rect = self.image.get_rect()
        self.pacman = pacman

        self.direction = random.choice(Constants.DIRECTION)
        # state is of type integer; where
        # 0 is 'NOT_CHASE'; that is ghost is moving in random direction
        # 1 is 'CHASE'; that is ghost is moving towards a goal position
        # 2 is 'BEING_CHASE'; that is ghost can be eaten by pacman/being chased
        # 3 is 'DEAD'; that is ghost is dead/eaten by pacman
        self.state = 0
        self.ghost_color = ()

        # For movement
        self.frame_counter = 0
        self.speed_difference = 1    # This is how much frames the ghost is slower than pacman (increase to slower ghost speed)

        # Astar components
        self.astar_helper = HelperFunction.AStar(board.map)
        self.path = []
        self.goal_pos = HelperFunction.current_position(self.pacman)

        # Start the state switch timer thread
        self.state_switch_timer = 10        # this is the timer for changing ghost state from 'NOT_CHASE' to 'CHASE' and vice versa; should be different for each ghost
        self.being_chased_timer = 10        # how long ghost will be vulnerable; 'BEING_CHASED'
        self.death_timer = 5                # how long the ghost will remain dead
        self.timer_thread = threading.Thread(target=self.run_timer)
        self.timer_thread.daemon = True
        self.timer_thread.start()


    def run_timer(self):
        small_sleep_interval = 0.1
        while True:
            if self.state < 2:
                elapsed_time = 0
                while self.state < 2 and elapsed_time < self.state_switch_timer:
                    time.sleep(small_sleep_interval)
                    elapsed_time += small_sleep_interval
                if self.state < 2:
                    self.state = 1 - self.state
                print(f"State changed to: {Constants.GHOST_STATE[self.state]}")
            elif self.state == 2:
                print('Ghost running')
                elapsed_time = 0
                while self.state == 2 and elapsed_time < self.being_chased_timer:
                    time.sleep(small_sleep_interval)
                    elapsed_time += small_sleep_interval
                if self.state == 2:
                    print('Ghost running completed')
                    self.update_state(0)
            else:
                elapsed_time = 0
                while self.state == 3 and elapsed_time < self.death_timer:
                    time.sleep(small_sleep_interval)
                    elapsed_time += small_sleep_interval
                if self.state == 3:
                    self.spawn_ghost()
            print(f"State changed to: {Constants.GHOST_STATE[self.state]}")

    
    def update(self):
        # Updating frame
        self.frame_counter += 1
        if self.frame_counter >= Constants.MOVE_DELAY + self.speed_difference:
            self.frame_counter = 0
            self.goal_pos = self.goal_update()

            # Movement
            C_GhostMovement.update(self= self)

    def draw(self, screen):                
        # # Ghost is not dead
        if  self.state != 3:
            # Ghost is not being chased
            if self.state == 2:
                self.ghost_color = Constants.GREY
            else:
                self.ghost_color = Constants.RED
            PG.draw.rect(screen, self.ghost_color, PG.Rect(self.rect.x * Constants.BOARD_SIZE, 
                                                           self.rect.y * Constants.BOARD_SIZE, 
                                                           Constants.BOARD_SIZE, Constants.BOARD_SIZE))

    def dead_position(self):
        self.rect.x = -Constants.GHOST_SIZE
        self.rect.y = -Constants.GHOST_SIZE
        # refreshing dead counter 
        self.death_counter = 0
        
    # Updating ghost state 
    def update_state(self, index):
        self.state = index
        # checking for dead state
        if self.state == 3:
            self.dead_position()

    def current_state(self):
        return self.state
    
    def spawn_ghost(self):
        # Updating ghost state
        self.update_state(0)
        # Updating ghost position to initial position
        self.rect.x, self.rect.y = self.initialx, self.initialy
        
    def goal_update(self):
        new_goal = ()
        # Ghost can chase/not chase
        if self.state < 2:
            new_goal = HelperFunction.current_position(self.pacman)     # this will be different (chasing position) for each ghost?

        # Ghost is being chased
        else:
            new_goal = (self.initialx, self.initialy)
        return new_goal
    
    def random_movement(self):
        # Current position
        ghost_x, ghost_y = self.rect.x, self.rect.y

        # Determine the next position based on current direction
        if self.direction == 'LEFT':
            ghost_x -= 1
        elif self.direction == 'RIGHT':
            ghost_x += 1
        elif self.direction == 'UP':
            ghost_y -= 1
        elif self.direction == 'DOWN':
            ghost_y += 1

        # Check if the new position is valid
        if HelperFunction.can_move(self, ghost_x, ghost_y):
            # Move to the new position
            self.rect.x = ghost_x
            self.rect.y = ghost_y
        # else:
        #     # If movement is blocked, do not change direction
        #     return

        # Decide whether to change direction based on a lower probability
        if random.random() < 0.1:
            new_direction = random.choice(Constants.DIRECTION)
            
            # Prevent moving directly back to the previous direction
            if (self.direction == 'LEFT' and new_direction == 'RIGHT') or \
            (self.direction == 'RIGHT' and new_direction == 'LEFT') or \
            (self.direction == 'UP' and new_direction == 'DOWN') or \
            (self.direction == 'DOWN' and new_direction == 'UP'):
                return
            
            # Update the direction
            self.direction = new_direction