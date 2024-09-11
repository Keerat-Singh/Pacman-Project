import numpy as np

# Creating Segment Tree

class SegmentTree():
    def __init__(self, capacity, operation, neutral_value):
        """
        Initialize the segment tree.

        Args:
            capacity (int): The size of the segment tree (must be a power of 2).
            operation (callable): The operation to be applied (e.g., np.add for sum, np.min for min).
            neutral_value: The value used for neutral elements (e.g., 0 for sum, infinity for min).
        """
        self.capacity = capacity
        self.operation = operation
        self.neutral_value = neutral_value
        # self.tree = [neutral_value] * (2 * capacity)  # Initialize with neutral values
        self.tree = [neutral_value for _ in range(2 * capacity)]  # Initialize with neutral values

    def _operate_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )
    
    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)
    

    def __setitem__(self, idx, value):
        """
        Set the value at index `idx` in the tree.

        Args:
            idx (int): The index to set the value at.
            value (float): The value to set.
        """

         # Ensure index is valid
        if not 0 <= idx < self.capacity:
            raise IndexError(f"Index {idx} is out of bounds for capacity {self.capacity}")
        
        # Leaf index in the tree
        idx += self.capacity
        self.tree[idx] = value
        idx //= 2

        # Update parent nodes
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        """
        Get the value at index `idx`.

        Args:
            idx (int): The index to retrieve.
        """
        assert (0 <= idx < self.capacity), "IDX is going negative or out of bounds"
        # print(f"IDX: {idx}")
        return self.tree[self.capacity + idx]


# Creating sum and min segment tree

class SumSegmentTree():

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = SegmentTree(capacity= capacity, operation= np.add, neutral_value= 0)  # Use numpy add for sum operation

    def store(self, idx, value):
        # Set the value at index 'idx'
        self.tree[idx] = value

    def sum(self, start:int = 0, end:int = 0):
        # Query the sum in range [start, end]
        return self.tree.operate(start, end)

    def retrieve(self, upperbound: float):
        """Find the highest index `i` about upper bound in the tree"""
        idx = 1
        # print(f"Capacity: {self.capacity}")
        while 2*idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = left  # Move to the right child
            else:
                upperbound -= self.tree[left]
                idx = right
            
            # print(f"IDX value after loop iteration: {idx}")
        return 2*idx - self.capacity  # Convert tree index to array index
    

class MinSegmentTree:
    def __init__(self, capacity):
        # Ensure capacity is a power of 2 for easy indexing
        self.capacity = capacity
        self.tree = SegmentTree(capacity= capacity, operation= min, neutral_value= float('inf'))  # Use Python's min function for minimum operation

    def min(self, start:int = 0, end:int = 0):
        # Query the minimum in range [start, end]
        return self.tree.operate(start, end)