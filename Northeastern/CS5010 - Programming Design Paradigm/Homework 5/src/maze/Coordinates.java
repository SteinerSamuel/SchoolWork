package maze;

import java.util.Objects;

/**
 * Coordinate class to hold positions.
 */
public class Coordinates {
  int xCoordinates;
  int yCoordinates;

  /**
   * Default constructor for a coordinate.
   *
   * @param x the x coord
   * @param y the y coord
   */
  public Coordinates(int x, int y) {
    this.xCoordinates = x;
    this.yCoordinates = y;
  }

  /**
   * Gets the x coordinate.
   *
   * @return the x coordinate value
   */
  public int getxCoordinates() {
    return xCoordinates;
  }

  /**
   * Set the x Coordinate.
   *
   * @param xCoordinates the new x coord
   */
  public void setxCoordinates(int xCoordinates) {
    this.xCoordinates = xCoordinates;
  }

  /**
   * Gets the y coordinate.
   *
   * @return the y coordinate value
   */
  public int getyCoordinates() {
    return yCoordinates;
  }

  /**
   * Set the y coordinate.
   *
   * @param yCoordinates the y coord
   */
  public void setyCoordinates(int yCoordinates) {
    this.yCoordinates = yCoordinates;
  }

  @Override
  public String toString() {
    return String.format("(%d, %d)", xCoordinates, yCoordinates);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    Coordinates that = (Coordinates) o;
    return xCoordinates == that.xCoordinates
            && yCoordinates == that.yCoordinates;
  }

  @Override
  public int hashCode() {
    return Objects.hash(xCoordinates, yCoordinates);
  }
}
