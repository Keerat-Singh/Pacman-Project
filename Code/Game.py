import pygame as PG
import random
import Constants
from Ghost import C_Ghost
from Pacman import C_Pacman

# Initialize PG
PG.init()

# Screen setup
screen = PG.display.set_mode((Constants.SCREEN_WIDTH, Constants.SCREEN_HEIGHT))
PG.display.set_caption("Pac-Man Game")

# Main loop
def main():
    clock = PG.time.Clock()
    all_sprites = PG.sprite.Group()
    ghosts = PG.sprite.Group()

    pacman = C_Pacman()
    all_sprites.add(pacman)

    for _ in range(Constants.NUMBER_OF_GHOST):
        ghost = C_Ghost()
        all_sprites.add(ghost)
        ghosts.add(ghost)

    running = True
    while running:
        for event in PG.event.get():
            if event.type == PG.QUIT:
                running = False

        all_sprites.update()

        # Check for collisions between Pac-Man and ghosts
        if PG.sprite.spritecollideany(pacman, ghosts):
            running = False  # End the game on collision

        screen.fill(Constants.BLACK)
        all_sprites.draw(screen)
        PG.display.flip()

        clock.tick(Constants.FPS)

    PG.quit()

if __name__ == "__main__":
    main()