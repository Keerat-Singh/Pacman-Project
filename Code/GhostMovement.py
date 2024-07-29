import pygame as PG
import Constants
import random
import HelperFunction

class C_GhostMovement():

    def update(self):
        
        # Will check for different ghost state and update movement accourdingly
        match self.state:
            case 0:

                HelperFunction.random_movement(self)
                
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

                # ghost_pos = (self.rect.x, self.rect.y)
                # # print(f"Current Ghost position: {ghost_pos}")
                # self.path = self.astar_helper.find_path(ghost_pos, self.goal_pos)
                # if self.path:
                #     next_pos = self.path.pop(0)
                #     # if self.can_move(next_pos[0], next_pos[1]):
                #     if HelperFunction.can_move(self, next_pos[0], next_pos[1]):
                #         self.rect.x = next_pos[0]
                #         self.rect.y = next_pos[1]

                ghost_pos = (self.rect.x, self.rect.y)
                pacman_pos = HelperFunction.current_position(self.pacman)
                initial_pos = (self.initialx, self.initialy)
                
                if HelperFunction.is_pacman_closer(self, pacman_pos, initial_pos):
                    HelperFunction.run_away_from_pacman(self, pacman_pos)
                else:
                    self.path = self.astar_helper.find_path_avoiding_pacman(ghost_pos, initial_pos, pacman_pos)
                    if self.path:
                        next_pos = self.path.pop(0)
                        if HelperFunction.can_move(self, next_pos[0], next_pos[1]):
                            self.rect.x = next_pos[0]
                            self.rect.y = next_pos[1]
            
            # Can remove below states
            case 3:
                return 'ghost is dead'
            case _:
                return "a new state found"