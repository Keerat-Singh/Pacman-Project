from Ghost import C_Ghost
import Constants
import HelperFunction

class C_Clyde(C_Ghost):
    def __init__(self, board, pacman):
        super().__init__(board, pacman)

        self.rect.x, self.rect.y = Constants.INITIAL_CLYDE_POSITION[Constants.MAP_INDEX]
        self.initialx, self.initialy = Constants.INITIAL_CLYDE_POSITION[Constants.MAP_INDEX]
        self.ghost_color = Constants.CLYDE_COLOR

        # looping path; we are caculating the smallest path when ghosts are generated; only once         
        self.smallest_loop_path = self.smallest_looping_path(self.initialx, self.initialy, board.map)
        print(f"Smallest loop path for Clyde: {self.smallest_loop_path}")

        # chasing
        self.chasing = True

    def goal_update(self):

        # Updaing Clyde chasing so it will go back home and not stuck in up and down chase
        if not self.chasing:
            if (self.rect.x, self.rect.y) == (self.initialx, self.initialy):
                self.chasing = True

        if self.state < 2 and self.chasing:
            pacman_pos = HelperFunction.current_position(self.pacman)
            ghost_pos = (self.rect.x, self.rect.y)
            if HelperFunction.calculate_distance(ghost_pos, pacman_pos) > 8:
                return pacman_pos
            else:
                self.chasing = False
                return (self.initialx, self.initialy)
        else:
            return (self.initialx, self.initialy)