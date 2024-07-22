import Constants

def AStar():
    pass

def can_move(self, new_x, new_y):
    # Check if the new position is within the screen boundaries
    if new_x < 0 or new_x >= len(self.board.map[0]) or new_y < 0 or new_y >= len(self.board.map):
        return False
    # Check if the new position is a valid cell (0)
    return self.board.map[new_y][new_x] != 1