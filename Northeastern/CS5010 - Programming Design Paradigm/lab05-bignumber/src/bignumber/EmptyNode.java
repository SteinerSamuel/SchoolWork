package bignumber;

/**
 * An empty node used as a stop marker or an empty Big number of value 0.
 **/
public class EmptyNode implements LittleEndianN {
  final int VALUE = 0;

  @Override
  public int getLength() {
    return 0;
  }

  @Override
  public LittleEndianN shiftLeft() {
    return this;
  }

  @Override
  public LittleEndianN shiftRight() {
    return this;
  }

  @Override
  public String toString() {
    return "";
  }

  @Override
  public LittleEndianN add(int value) {
    if (value > 9) {
      throw new IllegalArgumentException("Value must be single digit!");
    } else if (value < 0) {
      throw new IllegalArgumentException("Value must be non-negative");
    }
    return new LittleEndianNode(value, this);
  }

  @Override
  public int getValue() {
    return 0;
  }

  @Override
  public int valueAt(int pos) {
    throw new IllegalArgumentException("The position must be between 0 and the length of the number"
            + " minus 1 inclusive");
  }

  @Override
  public LittleEndianN getRest() {
    return new EmptyNode();
  }

  @Override
  public LittleEndianN addNode(LittleEndianN node, int carryover) {
    return new LittleEndianNode(node.getValue() + carryover, node.getRest());
  }
}
