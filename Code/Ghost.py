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

    # TODO the sleep timer is running for check if state is 0 or 1; even when the state has been changed to 2 or 3
    def run_timer(self):
        while True:
            
            time.sleep(self.state_switch_timer)
            # if ghost state can switch and is 0 or 1 then change to 0 or 1
            if self.state < 2:
                self.state = 1 - self.state
            elif self.state == 2:
                print('Ghost running')
                time.sleep(self.being_chased_timer)
                print('Ghost running completed')
                self.update_state(0)
            else:
                time.sleep(self.death_timer)
                self.spawn_ghost()
            print(f"State changed to: {Constants.GHOST_STATE[self.state]}")
                # self.update_state(0)
            # self.toggle_state()


    def toggle_state(self):
        if self.state == 0:
            self.state = 1
        elif self.state == 1:
            self.state = 0
        print(f"State changed to: {Constants.GHOST_STATE[self.state]}")
        
    
    def update(self):
        # Timer to switing state
        # if self.state_switch_counter < self.state_switch_timer:
        #     self.state = 1 - self.state
        #     print("How fast is switch: ", Constants.GHOST_STATE[self.state])
        #     self.state_switch_counter = 0

        # Updating frame
        self.frame_counter += 1
        if self.frame_counter >= Constants.MOVE_DELAY + Constants.SPEED_DIFFERENCE:
            self.frame_counter = 0
            self.goal_pos = self.goal_update()
            # Updating values to update ghost state from 0 to 1 and vice versa

            # match self.state:
            #     case 0:
            #         return
            #     # update goal_pos for AStar algo
            #     case 1:
            #         self.goal_pos = HelperFunction.current_position(self.pacman)
            #     case 2:
            #         self.goal_pos = (self.initialx, self.initialy)
                
            # pacman_pos = HelperFunction.current_position(self.pacman)
            # # print(f"Current Pacman position: {pacman_pos}")
            # ghost_pos = (self.rect.x, self.rect.y)
            # # print(f"Current Ghost position: {ghost_pos}")
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