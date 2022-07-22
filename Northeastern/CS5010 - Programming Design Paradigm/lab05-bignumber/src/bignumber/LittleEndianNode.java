package bignumber;

/**
 * A little endian node for a linked list.
 */
public class LittleEndianNode implements LittleEndianN {
  int value;
  LittleEndianN rest;

  /**
   * Constructor for a little endian node.
   *
   * @param value the value of this node
   * @param rest  the following node
   */
  public LittleEndianNode(int value, LittleEndianN rest) {
    this.value = value;
    this.rest = rest;
  }

  @Override
  public int getLength() {
    return 1 + rest.getLength();
  }

  @Override
  public LittleEndianN shiftLeft() {
    return new LittleEndianNode(0, this);
  }

  @Override
  public LittleEndianN shiftRight() {
    return this.rest;
  }

  @Override
  public LittleEndianN add(int value) {
    if (value > 9) {
      throw new IllegalArgumentException("Value must be single digit!");
    } else if (value < 0) {
      throw new IllegalArgumentException("Value must be non-negative");
    } else if (this.value + value > 10) {
      int upper = (this.value + value) / 10;
      int lower = (this.value + value) % 10;

      return new LittleEndianNode(lower, rest.add(upper));
    } else {
      return new LittleEndianNode(this.value + value, rest);
    }
  }

  @Override
  public int getValue() {
    return value;
  }

  @Override
  public int valueAt(int pos) {
    if (pos == 0) {
      return getValue();
    } else {
      return rest.valueAt(pos - 1);
    }
  }

  @Override
  public String toString() {
    return rest.toString() + value;
  }

  @Override
  public LittleEndianN getRest() {
    return rest;
  }

  @Override
  public LittleEndianN addNode(LittleEndianN node, int carryover) {

    int u = (this.value + node.getValue() + carryover) / 10;
    int l = (this.value + node.getValue() + carryover) % 10;
    return new LittleEndianNode(l, rest.addNode(node.getRest(), u));
  }
}
