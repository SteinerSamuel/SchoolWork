package model;

/**
 * A class which holds a data struct for a wall has no public methods besides to default.
 */
public final class Wall {
  private final Coordinates position;
  private final Coordinates deltaPosition;
  private  final CardinalDirections dir;

  /**
   * Constructor of a wall.
   *
   * @param y   The y co-ord of the starting node
   * @param x   the x co-ord of the starting node
   * @param dy  the y co-ord of the neighbor node
   * @param dx  the x co-ord of the neighbor node
   * @param dir the direction the nodes are neighbors from the starting node
   */
  public Wall(int x, int y, int dx, int dy, CardinalDirections dir) {
    this.position = new Coordinates(x, y);
    this.deltaPosition = new Coordinates(dx, dy);
    this.dir = dir;
  }

  public CardinalDirections getDir() {
    return dir;
  }

  public Coordinates getPosition() {
    return position;
  }

  public Coordinates getDeltaPosition() {
    return deltaPosition;
  }
}
