import pygame as PG
import Constants
import random
import HelperFunction

class C_GhostMovement():

    def update(self):
        
        # Will check for different ghost state and update movement accourdingly
        match self.state:
            case 0:

                # Calculate the potential new position based on input
                ghost_x, ghost_y = self.rect.x, self.rect.y
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
                if HelperFunction.can_move(self, ghost_x, ghost_y):
                    self.rect.x = ghost_x
                    self.rect.y = ghost_y
                
            case 1:

                ghost_pos = (self.rect.x, self.rect.y)
                # print(f"Current Ghost position: {ghost_pos}")
                self.path = self.astar_helper.find_path(ghost_pos, self.goal_pos)
                if self.path:
                    next_pos = self.path.pop(0)
                    # if self.can_move(next_pos[0], next_pos[1]):
                    if HelperFunction.can_move(self, next_pos[0], next_pos[1]):
                        self.rect.x = next_pos[0]
                        self.rect.y = next_pos[1]

            case 2:

                ghost_pos = (self.rect.x, self.rect.y)
                # print(f"Current Ghost position: {ghost_pos}")
                self.path = self.astar_helper.find_path(ghost_pos, self.goal_pos)
                if self.path:
                    next_pos = self.path.pop(0)
                    # if self.can_move(next_pos[0], next_pos[1]):
                    if HelperFunction.can_move(self, next_pos[0], next_pos[1]):
                        self.rect.x = next_pos[0]
                        self.rect.y = next_pos[1]
            
            # Can remove below states
            case 3:
                return 'ghost is dead'
            case _:
                return "a new state found"