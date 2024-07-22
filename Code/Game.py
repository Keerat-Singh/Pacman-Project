import pygame as PG
import random
import Constants
from Ghost import C_Ghost
from Pacman import C_Pacman
import time
from Board import C_Board

# Initialize PG
PG.init()

# Initializing board- this currently setups screen height/width and gets len/width of board
board = C_Board(Constants.MAP_INDEX)

# Screen setup
screen = PG.display.set_mode((Constants.screen_width, Constants.screen_height))
PG.display.set_caption("Pac-Man Game")

# TODO Might need to put this under tile.py (after creating)? cuz need to check for food/power up collision/eat
def draw_board(screen, board):
    for y, row in enumerate(board.map):
        # print(f"value of Y: {y}")
        for x, cell in enumerate(row):
            # print(f"value of X: {x}")
            if cell == 1:
                color = Constants.BLUE
            elif cell == 6:
                color = (122,122,122)       # Need to update color info
            else:
                continue  # Ignore other values
            PG.draw.rect(screen, color, PG.Rect(x * Constants.BOARD_SIZE, y * Constants.BOARD_SIZE, Constants.BOARD_SIZE, Constants.BOARD_SIZE))

# Main loop
def main(board):

    # Initializing 
    clock = PG.time.Clock()

    # Creating pacman object
    pacman = C_Pacman(board)
    
    # Creating ghosts object which will be changed to each specific ghost later
    ghosts = [C_Ghost(board) for _ in range(Constants.NUMBER_OF_GHOST)]

    RUNNING = True
    while RUNNING:

        for event in PG.event.get():
            if event.type == PG.QUIT:
                RUNNING = False

        pacman.update()
        for ghost in ghosts:
            ghost.update()

        # Check for collisions between Pac-Man and ghosts
        if PG.sprite.spritecollideany(pacman, ghosts):
            RUNNING = False  # End the game on collision

        screen.fill(Constants.BLACK)

        # Draw the board
        draw_board(screen, board)

        # Draw Pacman
        pacman.draw(screen)

        # Draw ghosts; this will later be for each single ghost differently
        for ghost in ghosts:
            ghost.draw(screen)


        PG.display.flip()

        clock.tick(Constants.FPS)

    PG.quit()

if __name__ == "__main__":
    main(board)