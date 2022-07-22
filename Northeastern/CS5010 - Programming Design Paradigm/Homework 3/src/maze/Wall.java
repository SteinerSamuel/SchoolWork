package maze;

/**
 * A class which holds a data struct for a wall has no public methods besides to default.
 */
public final class Wall {
  int xpos;
  int ypos;
  int dx;
  int dy;
  CardinalDirections dir;

  /**
   * Constructor of a wall.
   *
   * @param y   The y co-ord of the starting node
   * @param x   the x co-ord of the starting node
   * @param dy  the y co-ord of the neighbor node
   * @param dx  the x co-ord of the neighbor node
   * @param dir the direction the nodes are neighbors from the starting node
   */
  public Wall(int y, int x, int dy, int dx, CardinalDirections dir) {
    this.xpos = x;
    this.ypos = y;
    this.dx = dx;
    this.dy = dy;
    this.dir = dir;
  }

  @Override
  public String toString() {
    return "Wall{" + "x=" + xpos + ", y=" + ypos + ", dx=" + dx + ", dy=" + dy + '}';
  }
}
