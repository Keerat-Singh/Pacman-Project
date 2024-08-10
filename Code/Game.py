import pygame as PG
import random
import Constants
from Ghost import C_Ghost
from Pacman import C_Pacman
import time
from Board import C_Board
import HelperFunction
from Blinky import C_Blinky
from Clyde import C_Clyde
from Inky import C_Inky
from Pinky import C_Pinky

# Initialize PG
PG.init()

# Creating a font object
font = PG.font.Font(None, 36)

# FUNCTIONS
# drawing board and dots
def draw_board(screen, board):
    for y, row in enumerate(board.map):
        for x, cell in enumerate(row):
            # '1' is wall
            if cell == 1:
                color = Constants.BLUE       
                PG.draw.rect(screen, color, PG.Rect(x * Constants.BOARD_SIZE, y * Constants.BOARD_SIZE, Constants.BOARD_SIZE, Constants.BOARD_SIZE))
            # '0' is dot
            elif cell == 0:
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
def display_score(screen):
    score_text = font.render(f'Score: {Constants.total_score}', True, Constants.WHITE)
    screen.blit(score_text, (10,10))

# Title Screen
def draw_title_screen(screen):
    screen.fill(Constants.BLACK)
    title_font = PG.font.Font(None, 72)
    title_text = title_font.render('Pac-Man', True, Constants.WHITE)
    start_font = PG.font.Font(None, 36)
    start_text = start_font.render('Press ENTER to Start', True, Constants.WHITE)
    
    screen.blit(title_text, (Constants.screen_width // 2 - title_text.get_width() // 2, Constants.screen_height // 2 - title_text.get_height() // 2 - 100))
    screen.blit(start_text, (Constants.screen_width // 2 - start_text.get_width() // 2, Constants.screen_height // 2 - start_text.get_height() // 2 - 50))

# End Screen
def draw_end_screen(screen, game_condition):
    screen.fill(Constants.BLACK)
    end_font = PG.font.Font(None, 72)
    end_text = end_font.render(game_condition, True, Constants.WHITE)
    score_font = PG.font.Font(None, 36)
    score_text = score_font.render(f'Final Score: {Constants.total_score}', True, Constants.WHITE)
    
    screen.blit(end_text, (Constants.screen_width // 2 - end_text.get_width() // 2, Constants.screen_height // 2 - end_text.get_height() // 2- 100))
    screen.blit(score_text, (Constants.screen_width // 2 - score_text.get_width() // 2, Constants.screen_height // 2 - score_text.get_height() // 2 - 50))
    
    # Draw Retry button
    retry_button = PG.Rect(Constants.screen_width // 2 - 100, Constants.screen_height // 2 + 50, 200, 50)
    PG.draw.rect(screen, Constants.WHITE, retry_button)
    retry_text = PG.font.Font(None, 36).render('Retry', True, Constants.BLACK)
    screen.blit(retry_text, (retry_button.x + retry_button.width // 2 - retry_text.get_width() // 2, retry_button.y + retry_button.height // 2 - retry_text.get_height() // 2))
    
    # Draw Exit button
    exit_button = PG.Rect(Constants.screen_width // 2 - 100, Constants.screen_height // 2 + 110, 200, 50)
    PG.draw.rect(screen, Constants.WHITE, exit_button)
    exit_text = PG.font.Font(None, 36).render('Exit', True, Constants.BLACK)
    screen.blit(exit_text, (exit_button.x + exit_button.width // 2 - exit_text.get_width() // 2, exit_button.y + exit_button.height // 2 - exit_text.get_height() // 2))
    
    return retry_button, exit_button

# Reseting the game
def reset_game(board, pacman, blinky, clyde, inky, pinky, ghosts):
    
    # Killling threads
    for ghost in ghosts:
        ghost.stop_timer()

    # Resetting values
    board = C_Board(Constants.MAP_INDEX)
    pacman = C_Pacman(board)
    blinky = C_Blinky(board, pacman)
    clyde = C_Clyde(board, pacman)
    inky = C_Inky(board, pacman, blinky)
    pinky = C_Pinky(board, pacman)
    ghosts = [blinky, clyde, inky, pinky]
    ghosts = ghosts[:Constants.NUMBER_OF_GHOST]
    Constants.total_score = 0

    return board, pacman, blinky, clyde, inky, pinky, ghosts


# Main loop
def main():

    # Initializing 
    clock = PG.time.Clock()

    # Game State
    game_state = 0

    # Initializing board- this currently setups screen height/width and gets len/width of board
    board = C_Board(Constants.MAP_INDEX)

    # Screen setup
    screen = PG.display.set_mode((Constants.screen_width, Constants.screen_height))
    PG.display.set_caption("Pac-Man Game")

    # Creating pacman object
    pacman = C_Pacman(board)
    
    # Creating ghosts object which will be changed to each specific ghost later; MAX 4?
    blinky = C_Blinky(board, pacman)
    clyde = C_Clyde(board, pacman)
    inky = C_Inky(board, pacman, blinky)
    pinky = C_Pinky(board, pacman)
    ghosts = [blinky, clyde, inky, pinky]
    ghosts = ghosts[:Constants.NUMBER_OF_GHOST]
    # Can add a limit using Constants.NUMBER_OF_GHOST

    RUNNING = True
    while RUNNING:

        # for event in PG.event.get():
        #     if event.type == PG.QUIT:
        #         RUNNING = False

        for event in PG.event.get():
            if event.type == PG.QUIT:
                RUNNING = False
            elif event.type == PG.KEYDOWN:
                if event.key == PG.K_RETURN:  # Press ENTER to start the game
                    if game_state == 0:
                        game_state = 1  # Switch to play state
                    elif game_state == 2:
                        RUNNING = False  # Exit the game from end screen
            elif event.type == PG.MOUSEBUTTONDOWN:
                if game_state == 2:
                    mouse_pos = event.pos
                    if retry_button.collidepoint(mouse_pos):
                        # Reseting game info and state
                        board, pacman, blinky, clyde, inky, pinky, ghosts = reset_game(board, pacman, blinky, clyde, inky, pinky, ghosts)
                        game_state = 1
                    elif exit_button.collidepoint(mouse_pos):
                        RUNNING = False  # Exit the game from end screen

        match game_state:

            # Title state
            case 0:
                draw_title_screen(screen)

            # Play state
            case 1:
                # Updating ghost position/info
                pacman.update()

                # Check for collisions between Pac-Man and ghosts
                for ghost in ghosts:
                    if HelperFunction.current_position(pacman) == HelperFunction.current_position(ghost):
                        # Get ghost current state to check if it can kill pacman or pacman can kill ghost
                        if ghost.current_state() < 2:
                            # Lose
                            game_state = 2
                        else:
                            # This will only happen when pacman has eaten the food
                            ghost.update_state(3)
                            Constants.total_score += Constants.GHOST_KILL_SCORE

                # Updating ghost position/info
                for ghost in ghosts:
                    ghost.update()
                    
                # Checking for food collision
                match board.map[HelperFunction.current_position(pacman)[1]][HelperFunction.current_position(pacman)[0]]:
                    case 0:
                        board.map[HelperFunction.current_position(pacman)[1]][HelperFunction.current_position(pacman)[0]] = 9
                        Constants.total_score += Constants.FOOD_SCORE
                        board.total_food_count -= 1
                    case 8:
                        board.map[HelperFunction.current_position(pacman)[1]][HelperFunction.current_position(pacman)[0]] = 9
                        Constants.total_score += Constants.POWER_UP_SCORE
                        board.total_food_count -= 1
                        for ghost in ghosts:
                            ghost.update_state(2)
                
                # If you have eaten food/power up -- win condition
                if board.total_food_count == 0:
                    # Win
                    game_state = 2

                screen.fill(Constants.BLACK)

                # Draw the board
                draw_board(screen, board)

                # Displaying score
                display_score(screen)

                # Draw Pacman
                pacman.draw(screen)

                # Draw ghosts; this will later be for each single ghost differently
                for ghost in ghosts:
                    ghost.draw(screen)
            
            # End screen state
            case 2:
                if board.total_food_count == 0:
                    game_condition = "Game Won"
                else:
                    game_condition = "Game Over"
                retry_button, exit_button = draw_end_screen(screen, game_condition)
            
            case _:
                return "ERROE: Out of bounds game state"
            
        PG.display.flip()

        clock.tick(Constants.FPS)

    PG.quit()

if __name__ == "__main__":
    main()