import pygame as PG
import sys
import os
import numpy as np
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Game import Constants
from Game import HelperFunction
from Game.Ghost import C_Ghost
from Game.Pacman import C_Pacman
from Game.Board import C_Board
from Game.Blinky import C_Blinky
from Game.Clyde import C_Clyde
from Game.Inky import C_Inky
from Game.Pinky import C_Pinky

import NN_Constants

class PacmanGame:

# INITIALIZING GAME
    def __init__(self):

        # Initializing
        PG.init()
        self.clock = PG.time.Clock()

        # Game State
        # Updated to start the game directly instead of going to the main screen
        self.game_state = 1

        # Initializing game objects
        self.board = C_Board(Constants.MAP_INDEX)
        self.pacman = C_Pacman(self.board)
        self.blinky = C_Blinky(self.board, self.pacman)
        self.clyde = C_Clyde(self.board, self.pacman)
        self.inky = C_Inky(self.board, self.pacman, self.blinky)
        self.pinky = C_Pinky(self.board, self.pacman)
        self.ghosts = [self.blinky, self.clyde, self.inky, self.pinky]
        self.ghosts = self.ghosts[:Constants.NUMBER_OF_GHOST]
        Constants.total_score = 0

        # Screen setup-- Constants.screen_width, Constants.screen_height these values are set after board has been initialized
        self.screen = PG.display.set_mode((Constants.screen_width, Constants.screen_height))
        PG.display.set_caption("Pac-Man Game")

        self.display()

    # used to reset game
    def reset_game(self):

        # Killling threads
        for ghost in self.ghosts:
            ghost.stop_timer()
        
        self.board = C_Board(Constants.MAP_INDEX)
        self.pacman = C_Pacman(self.board)
        self.blinky = C_Blinky(self.board, self.pacman)
        self.clyde = C_Clyde(self.board, self.pacman)
        self.inky = C_Inky(self.board, self.pacman, self.blinky)
        self.pinky = C_Pinky(self.board, self.pacman)
        self.ghosts = [self.blinky, self.clyde, self.inky, self.pinky]
        self.ghosts = self.ghosts[:Constants.NUMBER_OF_GHOST]
        Constants.total_score = 0

        # updaing game state
        self.update_game_state(1)

# DQN FUNCTIONS

    # Update the game state based on the action
    # Return next_state, reward, done (whether the game is over)
    def step(self, action):             # This is the main loop for our dqn 
        
        # updating pacman movement info
        self.pacman_update(action)

        # calculating reward when pacman is interacting with different objects in env
        reward = self.calculating_reward()
        # There are some updates that are also handled inside calculating reward function

        # Updating ghost info
        self.ghost_update()

        # get the state space after updates are done, that is next state space 
        next_state = self.get_state_space()

        # Checking if the game has been completed or not (that is when game state is not equal to 1)
        done = False if self.game_state == 1 else True

        return next_state, reward, done

    # Return the current state representation
    def get_state_space(self):
        
        # This should hold the wall/food info
        board_state = np.array(self.board.map).flatten()

        # Game state info (might be used to check for game end info)
        game_state = np.array([self.game_state])
        
        # Get Pac-Man's position
        pacman_position = np.array(HelperFunction.current_position(self.pacman))
        
        # Get the positions of the ghosts
        ghost_positions = []
        # ghost current states
        ghost_state = []
        for ghost in self.ghosts:
            ghost_positions.append(HelperFunction.current_position(ghost))
            ghost_state.append(ghost.state)
        ghost_positions = np.array(ghost_positions).flatten()
        ghost_state = np.array(ghost_state).flatten()
        
        # Combine the board state, Pac-Man's position, ghost positions and ghost state into a single state vector
        state = np.concatenate((board_state, game_state, pacman_position, ghost_positions, ghost_state))
        return state

    # Return a list or array of possible actions
    def get_action_space(self):
        
        # DIRECTION = ['LEFT', 'RIGHT', 'UP', 'DOWN'] and index 4 is stay
        action = [0,1,2,3]
        return action

# GAME FUNCTIONS
    # drawing board and dots
    def draw_board(self, screen, board):
        for y, row in enumerate(board.map):
            for x, cell in enumerate(row):
                # '1' is wall
                if cell == 1:
                    color = Constants.BLUE       
                    PG.draw.rect(screen, color, PG.Rect(x * Constants.BOARD_SIZE, y * Constants.BOARD_SIZE, Constants.BOARD_SIZE, Constants.BOARD_SIZE))
                # '3' is dot
                elif cell == 3:
                    color = Constants.YELLOW
                    PG.draw.circle(screen, color, (x * Constants.BOARD_SIZE + Constants.SIZE//2, y * Constants.BOARD_SIZE + Constants.SIZE//2), Constants.BOARD_SIZE // 4)
                # '8' is power up
                elif cell == 8:
                    color = Constants.GREEN
                    PG.draw.circle(screen, color, (x * Constants.BOARD_SIZE + Constants.SIZE//2, y * Constants.BOARD_SIZE + Constants.SIZE//2), Constants.BOARD_SIZE // 3)
                else:
                    # This is for when food/power up is eaten and we are not displaying
                    continue

    # Displaying score
    def display_score(self, screen):
        # Creating a font object
        font = PG.font.Font(None, 36)
        score_text = font.render(f'Score: {Constants.total_score}', True, Constants.WHITE)
        screen.blit(score_text, (10,10))

    # Updating game state
    def update_game_state(self, game_state):
        self.game_state = game_state

    # Calling all display functions from here
    def display(self):

        self.screen.fill(Constants.BLACK)
        # Draw the board
        self.draw_board(self.screen, self.board)
        # Displaying score
        self.display_score(self.screen)
        # Draw Pacman
        self.pacman.draw(self.screen)
        # Draw ghosts; this will later be for each single ghost differently
        for ghost in self.ghosts:
            ghost.draw(self.screen)

        PG.display.flip()

        self.clock.tick(Constants.FPS)    
 
    # Updating pacman movement info -- without any self.frame_counter
    def pacman_update(self, action):
        
        # self.pacman.frame_counter += 1
        # if self.pacman.frame_counter >= Constants.MOVE_DELAY:
        #     self.pacman.frame_counter = 0

            new_x, new_y = self.pacman.rect.x, self.pacman.rect.y
            # DIRECTION = ['LEFT', 'RIGHT', 'UP', 'DOWN'] and index 4 is stay
            if action == 0:  # LEFT
                new_x -= 1
                self.pacman.direction = 'LEFT'
            elif action == 1:  # RIGHT
                new_x += 1
                self.pacman.direction = 'RIGHT'
            elif action == 2:  # UP
                new_y += 1
                self.pacman.direction = 'UP'
            else:  # (action == 3) DOWN
                new_y -= 1
                self.pacman.direction = 'DOWN'
            
            # Check if the new position is valid and if it is return the new position
            if HelperFunction.can_move(self.pacman, new_x, new_y):
                self.pacman.rect.x = new_x
                self.pacman.rect.y = new_y

    def calculating_reward(self):

        # reward will range from 0-1
        reward = 0

        # Check for collisions between Pac-Man and ghosts
        for ghost in self.ghosts:
            if HelperFunction.current_position(self.pacman) == HelperFunction.current_position(ghost):
                # Get ghost current state to check if it can kill pacman or pacman can kill ghost
                if ghost.current_state() < 2:
                    # Lose
                    reward += NN_Constants.REWARDS['Death']
                    self.update_game_state(2)
                else:
                    # This will only happen when pacman has eaten the food
                    ghost.update_state(3)
                    Constants.total_score += Constants.GHOST_KILL_SCORE
                    reward += NN_Constants.REWARDS['Ghost Kill']

         # Checking for food collision
        if self.board.map[HelperFunction.current_position(self.pacman)[1]][HelperFunction.current_position(self.pacman)[0]] == 3:
            self.board.map[HelperFunction.current_position(self.pacman)[1]][HelperFunction.current_position(self.pacman)[0]] = 0
            Constants.total_score += Constants.FOOD_SCORE
            self.board.total_food_count -= 1
            reward += NN_Constants.REWARDS['Food'] + (self.board.INITIAL_TOTAL_FOOD - self.board.total_food_count)//100
        elif self.board.map[HelperFunction.current_position(self.pacman)[1]][HelperFunction.current_position(self.pacman)[0]] == 8:
            self.board.map[HelperFunction.current_position(self.pacman)[1]][HelperFunction.current_position(self.pacman)[0]] = 0
            Constants.total_score += Constants.POWER_UP_SCORE
            self.board.total_food_count -= 1
            reward += NN_Constants.REWARDS['Power Up'] + (self.board.INITIAL_TOTAL_FOOD - self.board.total_food_count)//100
            for ghost in self.ghosts:
                ghost.update_state(2)

        return reward

    def ghost_update(self):

        # TODO need to check if the ghost has threding enabled and need to add below code implementation
        # TODO Ghost food interaction, when ghost are in afraid state and pacman eats food, need to reset the timer for afraid  

        # self.frame_counter += 1
        # if self.frame_counter >= Constants.MOVE_DELAY + self.speed_difference:
        #     self.frame_counter = 0



        for ghost in self.ghosts:
            # ghost.update()
            ghost.goal_pos = ghost.goal_update()
            # Updating flag info; will always tell us if the ghost is at home location or not
            ghost.reached_home_flag = True if (ghost.rect.x, ghost.rect.y) == (ghost.initialx, ghost.initialy) else False

            # Movement
            # Will check for different ghost state and update movement accourdingly
            match ghost.state:
                case 0:
                    # ghost.speed_difference = 1
                    ghost.random_movement()
                    
                case 1:

                    # ghost.speed_difference = 1
                    ghost_pos = (ghost.rect.x, ghost.rect.y)
                    # print(f"Current Ghost position: {ghost_pos}")
                    ghost.path = ghost.astar_helper.find_path(ghost_pos, ghost.goal_pos)
                    if ghost.path:
                        next_pos = ghost.path.pop(0)
                        if HelperFunction.can_move(ghost, next_pos[0], next_pos[1]):
                            ghost.rect.x = next_pos[0]
                            ghost.rect.y = next_pos[1]

                case 2:

                    # ghost.speed_difference = 2
                    ghost_pos = (ghost.rect.x, ghost.rect.y)
                    pacman_pos = HelperFunction.current_position(ghost.pacman)
                    initial_pos = (ghost.initialx, ghost.initialy)

                    if ghost.reached_home_flag:
                        ghost.reached_home_case_2 = True

                    if ghost.reached_home_case_2:
                        ghost.looping_smallest_path()
                    else:
                        if HelperFunction.is_pacman_closer(ghost, pacman_pos, initial_pos):
                            HelperFunction.run_away_from_pacman(ghost, pacman_pos)
                        else:
                            ghost.path = ghost.astar_helper.find_path_avoiding_pacman(ghost_pos, initial_pos, pacman_pos)
                            if ghost.path:
                                next_pos = ghost.path.pop(0)
                                if HelperFunction.can_move(ghost, next_pos[0], next_pos[1]):
                                    ghost.rect.x = next_pos[0]
                                    ghost.rect.y = next_pos[1]
                
                case 3:
                    return 'ghost is dead'
                case _:
                    return "a new state found"