package model;

import java.util.Objects;

/**
 * Coordinate class to hold positions.
 */
public class Coordinates {
  private final int xcoordinates;
  private final int ycoordinates;

  /**
   * Default constructor for a coordinate.
   *
   * @param x the x coord
   * @param y the y coord
   */
  public Coordinates(int x, int y) {
    this.xcoordinates = x;
    this.ycoordinates = y;
  }

  /**
   * Gets the x coordinate.
   *
   * @return the x coordinate value
   */
  public int getXcoordinates() {
    return xcoordinates;
  }

  /**

  /**
   * Gets the y coordinate.
   *
   * @return the y coordinate value
   */
  public int getYcoordinates() {
    return ycoordinates;
  }

  @Override
  public String toString() {
    return String.format("(%d, %d)", xcoordinates, ycoordinates);
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
    return xcoordinates == that.xcoordinates
            && ycoordinates == that.ycoordinates;
  }

  @Override
  public int hashCode() {
    return Objects.hash(xcoordinates, ycoordinates);
  }
}
