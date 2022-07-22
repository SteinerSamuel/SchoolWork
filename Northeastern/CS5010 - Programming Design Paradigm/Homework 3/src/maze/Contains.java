package maze;

/**
 * Enum for what the node contains.
 */
public enum Contains {
  GOLD(1), THIEF(-1), NOTHING(0);

  private int value;

  Contains(int value) {
    this.value = value;
  }

  /**
   * Returns the value of the content this is just a weight.
   *
   * @return The weight value of the node.
   */
  public int getValue() {
    return value;
  }
}
