# Constants

# Default Components
DEFAULT_SIZE = 20
DEFAULT_SPEED = 3

# Size Component for each element
SIZE = 30

# Character Components
PACMAN_SIZE = SIZE
GHOST_SIZE = SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GREY = (120, 120, 120)

# Gameplay Components
NUMBER_OF_GHOST = 4             # From 1-4
PACMAN_SPEED = .2
GHOST_SPEED = 6
DIRECTION = ['LEFT', 'RIGHT', 'UP', 'DOWN']
# DIRECTION = ['LEFT', 'DOWN']
# PACMAN_STATE = ['ALIVE', 'CHASING', 'DEAD']
PACMAN_LIVE = 3
GHOST_STATE = ['NOT_CHASE', 'CHASE', 'BEING_CHASED', 'DEAD']
BOARD_SIZE = SIZE
MOVE_DELAY = 5
SPEED_DIFFERENCE = 0    # This is how much frames the ghost is slower than pacman (increase to slower ghost speed)

# Initial position
INITIAL_PACMAN_POSITION = [[15,10]]     # This should map with board and MAP_INDEX
INITIAL_BLINKY_POSITION = [[24,29]]     # This should map with board and MAP_INDEX
INITIAL_CLYDE_POSITION = [[24,1]]     # This should map with board and MAP_INDEX
INITIAL_INKY_POSITION = [[3,1]]     # This should map with board and MAP_INDEX
INITIAL_PINKY_POSITION = [[3,29]]     # This should map with board and MAP_INDEX

# Score
POWER_UP_SCORE = 50
FOOD_SCORE = 10
GHOST_KILL_SCORE = 500
TOTAL_SCORE = 0

# Game State
GAME_STATE = ['TITLE', 'PLAY', 'ENDSCREEN']
MAP_INDEX = 0

# Values which are updated in game
board_width = 0
board_height = 0

# Pygame Components
screen_width = 0
screen_height = 0
FPS = 30