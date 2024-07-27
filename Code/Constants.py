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
NUMBER_OF_GHOST = 1
PACMAN_SPEED = .2
GHOST_SPEED = 6
DIRECTION = ['LEFT', 'RIGHT', 'UP', 'DOWN']
# DIRECTION = ['LEFT', 'DOWN']
# PACMAN_STATE = ['ALIVE', 'CHASING', 'DEAD']
PACMAN_LIVE = 3
GHOST_STATE = ['NOT_CHASE', 'CHASE', 'BEING_CHASED', 'DEAD']
BOARD_SIZE = SIZE
MOVE_DELAY = 5
SPEED_DIFFERENCE = 3    # This is how much frames the ghost is slower than pacman
INITIAL_PACMAN_POSITION = [[15,10]]     # This should map with board and MAP_INDEX
# TODO INITIAL_GHOST_POSITION

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