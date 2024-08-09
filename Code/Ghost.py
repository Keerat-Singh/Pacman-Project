import pygame as PG
import random
import Constants
from GhostMovement import C_GhostMovement
import HelperFunction
import time
import threading
import math
import heapq

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

        # For movement
        self.frame_counter = 0
        self.speed_difference = 1    # This is how much frames the ghost is slower than pacman (increase to slower ghost speed)

        # Astar components
        self.astar_helper = HelperFunction.AStar(board.map)
        self.path = []
        self.goal_pos = HelperFunction.current_position(self.pacman)

        # Start the state switch timer thread
        self.state_switch_timer = 10        # this is the timer for changing ghost state from 'NOT_CHASE' to 'CHASE' and vice versa; should be different for each ghost
        self.to_chase_switch_timer = 7      # this is the timer for changing ghost state from 'NOT_CHASE' to 'CHASE'
        self.to_notChase_switch_timer = 20  # this is the timer for changing ghost state from 'CHASE' to 'NOT_CHASE'
        self.being_chased_timer = 10        # how long ghost will be vulnerable; 'BEING_CHASED'
        self.death_timer = 6                # how long the ghost will remain dead
        self.timer_thread = threading.Thread(target=self.run_timer)
        self.timer_thread.daemon = True
        self.timer_thread.start()

        # Reached home flag 
        self.reached_home_flag = False
        self.reached_home_case_2 = False

    def run_timer(self):
        small_sleep_interval = 0.1      # This is used in addition with elapsed time to have timer
        while True:
            # Ghost scatter/not chase state
            if self.state == 0:         
                elapsed_time = 0
                while self.state < 2 and elapsed_time < self.to_chase_switch_timer:
                    time.sleep(small_sleep_interval)
                    elapsed_time += small_sleep_interval
                # Checking again if the ghost state is indeed scatter/not chase
                if self.state == 0:
                    self.update_state(1)
            # Ghost chase state
            elif self.state == 1:
                elapsed_time = 0
                while self.state < 2 and elapsed_time < self.to_notChase_switch_timer:
                    time.sleep(small_sleep_interval)
                    elapsed_time += small_sleep_interval
                if self.state == 1:
                    self.update_state(0)
            # Ghost scared/run state
            elif self.state == 2:
                elapsed_time = 0
                while self.state == 2 and elapsed_time < self.being_chased_timer:
                    time.sleep(small_sleep_interval)
                    elapsed_time += small_sleep_interval
                if self.state == 2:
                    self.reached_home_case_2 = False
                    self.update_state(0)
            # Ghost dead state
            elif self.state == 3:
                elapsed_time = 0
                while self.state == 3 and elapsed_time < self.death_timer:
                    time.sleep(small_sleep_interval)
                    elapsed_time += small_sleep_interval
                if self.state == 3:
                    self.spawn_ghost()

    
    def update(self):
        # Updating frame
        self.frame_counter += 1
        if self.frame_counter >= Constants.MOVE_DELAY + self.speed_difference:
            self.frame_counter = 0
            self.goal_pos = self.goal_update()

            # Updating flag info; will always tell us if the ghost is at home location or not
            self.reached_home_flag = True if (self.rect.x, self.rect.y) == (self.initialx, self.initialy) else False

            # Movement
            C_GhostMovement.update(self= self)

    def draw(self, screen):                
        # # Ghost is not dead
        if  self.state != 3:
            # Ghost is not being chased
            if self.state == 2:
                color = Constants.GREY
            else:
                color = self.ghost_color
            PG.draw.rect(screen, color, PG.Rect(self.rect.x * Constants.BOARD_SIZE, 
                                                           self.rect.y * Constants.BOARD_SIZE, 
                                                           Constants.BOARD_SIZE, Constants.BOARD_SIZE))

    # Updating ghost state 
    def update_state(self, index):
        self.state = index
        # checking for dead state; and if dead update position to be out of bounds
        if self.state == 3:
            self.rect.x = -Constants.GHOST_SIZE
            self.rect.y = -Constants.GHOST_SIZE

    def current_state(self):
        return self.state
    
    def spawn_ghost(self):
        # Updating ghost state
        self.update_state(0)
        # Updating ghost position to initial position
        self.rect.x, self.rect.y = self.initialx, self.initialy
    
    def random_movement(self):
        # Current position
        ghost_x, ghost_y = self.rect.x, self.rect.y

        if random.random() < 0.1:
            self.direction = random.choice(Constants.DIRECTION)

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

        # # Decide whether to change direction based on a lower probability
        # if random.random() < 0.1:
        #     new_direction = random.choice(Constants.DIRECTION)
            
        #     # Prevent moving directly back to the previous direction
        #     if (self.direction == 'LEFT' and new_direction == 'RIGHT') or \
        #     (self.direction == 'RIGHT' and new_direction == 'LEFT') or \
        #     (self.direction == 'UP' and new_direction == 'DOWN') or \
        #     (self.direction == 'DOWN' and new_direction == 'UP'):
        #         return
            
        #     # Update the direction
        #     self.direction = new_direction


    def finding_smallest_looping_path(self, initialx, initialy, board):
        goal = (initialx, initialy) 
        visited = set()

        # this path will be returned which will inidicate the smallest loop for ghost
        valid_paths = []

        # Initially we are just checking two blocks away from the ghost and making middle block as visited (start - visited - initial position);
        # after this we will find the shortest path from start to initial position; and will also update valid path with this initial middle visited path info 
        start = self.getting_middle_path(initialx, initialy, valid_paths, visited)

        # Priority queue for A* (cost, current position, path)
        open_list = []
        heapq.heappush(open_list, (0 + HelperFunction.calculate_distance(start, goal), start, [start]))
        
        # Dictionaries for tracking costs and paths
        g_cost = {start: 0}
        f_cost = {start: HelperFunction.calculate_distance(start, goal)}
        
        while open_list:
            current_cost, current_pos, path = heapq.heappop(open_list)
            if current_pos in visited:
                continue
            
            visited.add(current_pos)
            
            # If the goal is reached, return the path
            if current_pos == goal:
                return valid_paths + path
            
            # Get neighbors
            x, y = current_pos
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            
            for nx, ny in neighbors:
                if self.is_valid_position(nx, ny, board, visited):
                    new_cost = g_cost[current_pos] + 1
                    if (nx, ny) not in g_cost or new_cost < g_cost[(nx, ny)]:
                        g_cost[(nx, ny)] = new_cost
                        f = new_cost + HelperFunction.calculate_distance((nx, ny), goal)
                        f_cost[(nx, ny)] = f
                        heapq.heappush(open_list, (f, (nx, ny), path + [(nx, ny)]))
                        
        return []  # Return empty path if no path is found
    

    def is_valid_position(self, x, y, board, visited):
        if (0 <= x < len(board[0])) and (0 <= y < len(board)) and (x, y) not in visited and board[y][x] != 1:
            return True
        return False
        

    def getting_middle_path(self, initialx, initialy, valid_paths, visited):
        x = initialx
        y = initialy

        # checking all the valid paths form the start position
        if HelperFunction.can_move(self, x - 1, y):   # Left
            x -= 1
        elif HelperFunction.can_move(self, x + 1, y):   # Right
            x += 1
        elif HelperFunction.can_move(self, x, y + 1):   # Up
            y += 1
        else:
            # Down 
            y -= 1

        # Updating path and visited info
        valid_paths.append((x,y))
        visited.add((x,y))

        # checking all the valid paths form the new x,y position 
        if HelperFunction.can_move(self, x - 1, y) and (x-1 != initialx):   # Left
            x -= 1
        elif HelperFunction.can_move(self, x + 1, y) and (x+1 != initialx):   # Right
            x += 1
        elif HelperFunction.can_move(self, x, y + 1) and (y+1 != initialy):   # Up
            y += 1
        else:
            # Down 
            y -= 1

        # x and y are the new starting position
        return (x, y)
    
    
    # This is to loop the smallest path for each respective ghost once they have reached their base
    def looping_smallest_path(self):
        
        if self.current_path_index < len(self.smallest_loop_path) - 1:
            self.rect.x, self.rect.y =  self.smallest_loop_path[self.current_path_index]
            self.current_path_index += 1
        else:
            # Loop back to the start of the path if needed
            self.current_path_index = 0
            self.rect.x, self.rect.y = self.smallest_loop_path[self.current_path_index]