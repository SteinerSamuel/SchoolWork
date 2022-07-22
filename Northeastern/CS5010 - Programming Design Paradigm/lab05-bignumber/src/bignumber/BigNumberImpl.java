package bignumber;

/**
 * Implementation of bigNumber interface.
 */
public class BigNumberImpl implements BigNumber {
  LittleEndianN startNode;

  /**
   * Default constructor makes a empty big number whose value is 0.
   */
  public BigNumberImpl() {
    this.startNode = new EmptyNode();
  }

  /**
   * Constructor which takes a string and turns it into a big number.
   *
   * @param number a string representation of a number.
   */
  public BigNumberImpl(String number) {
    number = number.replaceFirst("^0+(?!$)", "");
    this.startNode = new EmptyNode();
    for (char c : number.toCharArray()) {
      if (Character.isDigit(c)) {
        int value = Character.getNumericValue(c);
        this.startNode = new LittleEndianNode(value, this.startNode);
      } else {
        throw new IllegalArgumentException("You must provide a valid number!");
      }
    }
  }

  @Override
  public LittleEndianN getStartingNode() {
    return startNode;
  }

  @Override
  public int length() {
    return startNode.getLength();
  }

  @Override
  public BigNumber shiftLeft(int times) {
    if (times > 0) {
      for (int i = times; i > 0; i--) {
        this.startNode = this.startNode.shiftLeft();
      }
    } else {
      times = times * -1;
      for (int i = times; i > 0; i--) {
        this.startNode = this.startNode.shiftRight();
      }
    }
    return this;
  }

  @Override
  public BigNumber shiftRight(int times) {
    if (times > 0) {
      for (int i = times; i > 0; i--) {
        this.startNode = this.startNode.shiftRight();
      }
    } else {
      times = times * -1;
      for (int i = times; i > 0; i--) {
        this.startNode = this.startNode.shiftLeft();
      }
    }
    return this;
  }

  @Override
  public BigNumber addDigit(int digit) {
    if (digit > 9) {
      throw new IllegalArgumentException("Digit must be single digit!");
    } else if (digit < 0) {
      throw new IllegalArgumentException("Digit must be non negative!");
    }
    this.startNode = this.startNode.add(digit);
    return this;
  }

  @Override
  public int getDigitAt(int pos) {
    if (pos >= this.length()) {
      throw new IllegalArgumentException("The position must be in the number pos is bound by 0 and "
              + "the length of the number minus 1 inclusive");
    }
    return this.startNode.valueAt(pos);
  }

  @Override
  public BigNumber copy() {
    return new BigNumberImpl(this.toString());
  }


  @Override
  public BigNumber add(BigNumber number) {
    LittleEndianN node = this.startNode.addNode(number.getStartingNode(), 0);
    return new BigNumberImpl(node.toString());
  }

  @Override
  public String toString() {
    if (startNode instanceof EmptyNode) {
      return "0";
    }
    return startNode.toString();
  }

  @Override
  public int compareTo(BigNumber compare) {
    if (this.length() > compare.length()) {
      return 1;
    } else if (this.length() < compare.length()) {
      return -1;
    } else {
      for (int i = this.length() - 1; i >= 0; i--) {
        if (this.getDigitAt(i) > compare.getDigitAt(i)) {
          return 1;
        } else if (this.getDigitAt(i) < compare.getDigitAt(i)) {
          return -1;
        }
      }
      return 0;
    }
  }
}
