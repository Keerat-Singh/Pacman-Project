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

# Gameplay Components
NUMBER_OF_GHOST = 4
PACMAN_SPEED = .2
GHOST_SPEED = 6
DIRECTION = ['LEFT', 'RIGHT', 'UP', 'DOWN']
# DIRECTION = ['LEFT', 'DOWN']
PACMAN_STATE = ['ALIVE', 'CHASING', 'DEAD']
PACMAN_LIVE = 3
GHOST_STATE = ['CHASE', 'BEING_CHASED', 'DEAD']
POWER_UP_SCORE = 50
FOOD_SCORE = 10
GHOST_KILL_SCORE = 500
DISPLAY_SCORE = 0
BOARD_SIZE = SIZE
MOVE_DELAY = 10

# Power UPs
FOOD_COUNT = 50
POWER_UP_COUNT = 4

# Game State
GAME_STATE = ['TITLE', 'PLAY', 'ENDSCREEN']
MAP_INDEX = 0

# Values which are updated in game
board_width = 0
board_height = 0

# Pygame Components
screen_width = 0
screen_height = 0
FPS = 60