import numpy as np

class C_Node:

  # edges = {'Up': None, 'Down': None, 'Left': None, 'Right': None}
  # edges = np.array([[]])
  edges = []
  x = 0.0
  y = 0.0
  smallestDistToPointy = 10000000
  degree = 0
  value = 0
  checked = False

  def __init__(self, x1, y1):
    self.x = x1
    self.y = y1
  
  def show(self):
    pass
  
  def addEdges(self, nodes, tile):
    for i in len(nodes):
      if nodes[i].y == self.y or nodes[i].x == self.x:
        if nodes[i].y == self.y:
          mostLeft = min(nodes[i].x, self.x) +1
          maxLeft = max(nodes[i].x, self.x)
          edge = True
          while (mostLeft < maxLeft):
            if tile[int(self.y)][int(mostLeft)].wall:
              edge = False
              break
            mostLeft += 1
          if edge:
            # self.edges = np.append(self.edges, nodes[i], axis= 0)
            self.edges.append(nodes[i])

        elif nodes[i].x == self.x:
          mostUp = min(nodes[i].y, self.y) +1
          maxUp = max(nodes[i].y, self.y)
          edge = True
          while (mostUp < maxUp):
            if tile[int(mostUp)][int(self.x)].wall:
              edge = False
              break
            mostUp += 1
          if edge:
            # self.edges = np.append(self.edges, nodes[i], axis= 0)
            self.edges.append(nodes[i])

"""# -------------------------------------------------------------------------------------------
class C_Linked_List(C_Node):

  def __init__(self, x1=0, y1=0):
    self.x = x1
    self.y = y1

  def show():
    pass

  def addEdges(self, nodes):
    for i in len(nodes):
      pass
      # if odes[i

# --------------------------------------------------------------------------------------------"""

#-------------------------------------------------------------------------------------------------------------------------------------------------
#   //add all the nodes this node is adjacent to 
#   void addEdges(ArrayList<Node> nodes) {
#     for (int i =0; i < nodes.size(); i++) {//for all the nodes
#       if (nodes.get(i).y == y ^ nodes.get(i).x == x) {
#         if (nodes.get(i).y == y) {//if the node is on the same line horizontally
  #           float mostLeft = min(nodes.get(i).x, x) + 1;
  #           float max = max(nodes.get(i).x, x);
  #           boolean edge = true;
  #           while (mostLeft < max) {//look from the one node to the other looking for a wall
  #             if (tiles[(int)y][(int)mostLeft].wall) {
  #               edge = false;//not an edge since there is a wall in the way
  #               break;
  #             }
  #             mostLeft ++;//move 1 step closer to the other node
  #           }
  #           if (edge) {
  #             edges.add(nodes.get(i));//add the node as an edge
  #           }
#         } else if (nodes.get(i).x == x) {//same line vertically
#           float mostUp = min(nodes.get(i).y, y) + 1;
#           float max = max(nodes.get(i).y, y);
#           boolean edge = true;
#           while (mostUp < max) {
#             if (tiles[(int)mostUp][(int)x].wall) {
#               edge = false;
#               break;
#             }
#             mostUp ++;
#           }
#           if (edge) {
#             edges.add(nodes.get(i));
#           }
#         }
#       }
#     }
#   }


# class Node {

#   LinkedList<Node> edges = new LinkedList<Node>();//all the nodes this node is connected to 
#   float x;
#   float y;
#   float smallestDistToPoint = 10000000;//the distance of smallest path from the start to this node 
#   int degree;
#   int value;  
#   boolean checked = false;
#   //-------------------------------------------------------------------------------------------------------------------------------------------------
#   //constructor
#   Node(float x1, float y1) {
#     x = x1;
#     y = y1;
#   }
#   //-------------------------------------------------------------------------------------------------------------------------------------------------
#   //draw a litle circle
#   void show() {
#     fill(0, 100, 100);
#     ellipse(x*16 +8, y*16 +8, 10, 10 );
#   }

#   //-------------------------------------------------------------------------------------------------------------------------------------------------
#   //add all the nodes this node is adjacent to 
#   void addEdges(ArrayList<Node> nodes) {
#     for (int i =0; i < nodes.size(); i++) {//for all the nodes
#       if (nodes.get(i).y == y ^ nodes.get(i).x == x) {
#         if (nodes.get(i).y == y) {//if the node is on the same line horizontally
#           float mostLeft = min(nodes.get(i).x, x) + 1;
#           float max = max(nodes.get(i).x, x);
#           boolean edge = true;
#           while (mostLeft < max) {//look from the one node to the other looking for a wall
#             if (tiles[(int)y][(int)mostLeft].wall) {
#               edge = false;//not an edge since there is a wall in the way
#               break;
#             }
#             mostLeft ++;//move 1 step closer to the other node
#           }
#           if (edge) {
#             edges.add(nodes.get(i));//add the node as an edge
#           }
#         } else if (nodes.get(i).x == x) {//same line vertically
#           float mostUp = min(nodes.get(i).y, y) + 1;
#           float max = max(nodes.get(i).y, y);
#           boolean edge = true;
#           while (mostUp < max) {
#             if (tiles[(int)mostUp][(int)x].wall) {
#               edge = false;
#               break;
#             }
#             mostUp ++;
#           }
#           if (edge) {
#             edges.add(nodes.get(i));
#           }
#         }
#       }
#     }
#   }
# }