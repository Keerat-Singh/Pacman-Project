import pygame as PG
import Constants
import random
import HelperFunction

class C_GhostMovement():

    def update(self):
        
        # Will check for different ghost state and update movement accourdingly
        match self.state:
            case "NOT_CHASE":

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
                
            case "CHASE":

                # Astar will be called
                return 'test'
            case "BEING_CHASED":

                # Ghost will go to its initial position and then move randomly once reached 
                return 'test'
            
            # Can remove below states
            case "DEAD":
                return 'ghost is dead'
            case _:
                return "a new state found"