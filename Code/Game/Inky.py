from . import Constants
from . import HelperFunction
from .Ghost import C_Ghost

class C_Inky(C_Ghost):
    def __init__(self, board, pacman, blinky):
        super().__init__(board, pacman)
        
        self.blinky = blinky
        self.rect.x, self.rect.y = Constants.INITIAL_INKY_POSITION[Constants.MAP_INDEX]
        self.initialx, self.initialy = Constants.INITIAL_INKY_POSITION[Constants.MAP_INDEX]
        self.ghost_color = Constants.INKY_COLOR

        # looping path; we are caculating the smallest path when ghosts are generated; only once         
        self.smallest_loop_path = self.finding_smallest_looping_path(self.initialx, self.initialy, board.map)
        self.current_path_index = 0

    # TODO might need to update the target position logic as right now pinky can get stuck if target position is invalid; like a wall/out of bounds
    def goal_update(self):
        if self.state < 2:
            pacman_pos = HelperFunction.current_position(self.pacman)
            pacman_direction = self.pacman.direction
            if pacman_direction == 'UP':
                target_pos = (pacman_pos[0], pacman_pos[1] - 2)
            elif pacman_direction == 'DOWN':
                target_pos = (pacman_pos[0], pacman_pos[1] + 2)
            elif pacman_direction == 'LEFT':
                target_pos = (pacman_pos[0] - 2, pacman_pos[1])
            elif pacman_direction == 'RIGHT':
                target_pos = (pacman_pos[0] + 2, pacman_pos[1])
            
            blinky_pos = HelperFunction.current_position(self.blinky)
            inky_target_pos = (2 * target_pos[0] - blinky_pos[0], 2 * target_pos[1] - blinky_pos[1])
            return inky_target_pos
        else:
            return (self.initialx, self.initialy)