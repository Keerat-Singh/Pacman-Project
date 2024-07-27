import pygame as PG
import random
import Constants
from Ghost import C_Ghost
from Pacman import C_Pacman
import time
from Board import C_Board
from Food import C_Food
import HelperFunction

# Initialize PG
PG.init()

# Initializing board- this currently setups screen height/width and gets len/width of board
board = C_Board(Constants.MAP_INDEX)

# Screen setup
screen = PG.display.set_mode((Constants.screen_width, Constants.screen_height))
PG.display.set_caption("Pac-Man Game")

# Creating a font object
font = PG.font.Font(None, 36)

# Initializing food and power ups
temp = [[None]*(Constants.board_width) for i in range(Constants.board_height)]
for y, row in enumerate(board.map):
    for x, cell in enumerate(row):
        if cell == 0:
            temp[y][x] = C_Food(x, y, cell)
        elif cell == 8:
            temp[y][x] = C_Food(x, y, cell)

# Removing all the None values
foods = temp
# foods = [[value for value in row if value != None] for row in temp]
# foods=  [sublist for sublist in foods if sublist]

################################ Testing #################################################################################################

# print(Constants.INITIAL_PACMAN_POSITION[Constants.MAP_INDEX][0])
# print(Constants.INITIAL_PACMAN_POSITION[Constants.MAP_INDEX][1])

###########################################################################################################################################
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
                continue
                # color = Constants.WHITE       
                # PG.draw.rect(screen, color, PG.Rect(x * Constants.BOARD_SIZE, y * Constants.BOARD_SIZE, Constants.BOARD_SIZE, Constants.BOARD_SIZE))

# # Displaying food and power up
# def draw_food(screen, foods):
#     for y, row in enumerate((foods)):
#         for x, cell in enumerate(row):
#             if cell != None:
#                 cell.show(screen)

def display_score(screen):
    score_text = font.render(f'Score: {Constants.TOTAL_SCORE}', True, Constants.WHITE)
    screen.blit(score_text, (10,10))

# Main loop
def main():

    # Initializing 
    clock = PG.time.Clock()

    # Creating pacman object
    pacman = C_Pacman(board)
    
    # Creating ghosts object which will be changed to each specific ghost later; MAX 4?
    ghosts = [C_Ghost(board, pacman) for _ in range(Constants.NUMBER_OF_GHOST)]

    RUNNING = True
    while RUNNING:

        for event in PG.event.get():
            if event.type == PG.QUIT:
                RUNNING = False

        pacman.update()
        for ghost in ghosts:
            ghost.update()

        # Check for collisions between Pac-Man and ghosts
        for ghost in ghosts:
            print(ghost.state)
            if HelperFunction.current_position(pacman) == HelperFunction.current_position(ghost):
                # Get ghost current state to check if it can kill pacman or pacman can kill ghost
                # TODO Not working?? maybe cuz of astar?
                if ((ghost.current_state == "NOT_CHASE")):
                # if (ghost.current_state == 'NOT_CHASE'):
                    print("GAME ENDING!")
                    RUNNING = False  # End the game on collision
                else:
                    # This will only happen when pacman has eaten the food
                    # ghost.update_state(3)
                    # Constants.TOTAL_SCORE += Constants.GHOST_KILL_SCORE
                    print("Inside else: ",ghost.state)

        # if PG.sprite.spritecollideany(pacman, ghosts):

        # Checking for food collision
        match board.map[HelperFunction.current_position(pacman)[1]][HelperFunction.current_position(pacman)[0]]:
            case 0:
                board.map[HelperFunction.current_position(pacman)[1]][HelperFunction.current_position(pacman)[0]] = 9
                Constants.TOTAL_SCORE += Constants.FOOD_SCORE
            case 8:
                board.map[HelperFunction.current_position(pacman)[1]][HelperFunction.current_position(pacman)[0]] = 9
                Constants.TOTAL_SCORE += Constants.POWER_UP_SCORE
                for ghost in ghosts:
                    ghost.update_state(2)

        screen.fill(Constants.BLACK)

        # Draw the board
        draw_board(screen, board)

        # Draw the food and power up
        # draw_food(screen, foods)

        # Displaying score
        display_score(screen)

        # Draw Pacman
        pacman.draw(screen)

        # print(f"pacman position outside ghost info: {HelperFunction.current_position(pacman)}")

        # Draw ghosts; this will later be for each single ghost differently
        for ghost in ghosts:
            ghost.draw(screen)

        PG.display.flip()

        clock.tick(Constants.FPS)

    PG.quit()

if __name__ == "__main__":
    main()