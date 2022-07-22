package model;

/**
 * A enum for the the cardinal Directions this is used throughout the project for multiple reasons.
 */
public enum CardinalDirections {
  NORTH, SOUTH, EAST, WEST;

  /**
   * Returns the opposite direction of the current direction.
   *
   * @return the opposite direction of the current direction
   */
  public CardinalDirections opposite() {
    switch (this) {
      case NORTH:
        return CardinalDirections.SOUTH;
      case SOUTH:
        return CardinalDirections.NORTH;
      case EAST:
        return CardinalDirections.WEST;
      case WEST:
        return CardinalDirections.EAST;
      default:
        throw new IllegalStateException("Should not be possible");
    }
  }
}
