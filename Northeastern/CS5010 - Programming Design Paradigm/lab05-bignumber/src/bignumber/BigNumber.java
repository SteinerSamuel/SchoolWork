package bignumber;

/**
 * An interface for a big number this uses a little-endian approach to having big numbers.
 */
public interface BigNumber {
  /**
   * Calculates the number of digits in the number.
   *
   * @return The number of digits in the number.
   */
  int length();

  /**
   * Preforms a left bit shift operation on the big number a number of times.
   *
   * @param times the number of times to preform the operation
   * @return the new big number.
   */
  BigNumber shiftLeft(int times);

  /**
   * Performs a right bit shift operation o n the big number a number of times.
   *
   * @param times the number of times to preform the operation
   * @return the new big number
   */
  BigNumber shiftRight(int times);

  /**
   * Adds a digit to the current number.
   *
   * @param digit a non-negative single digit
   * @return the new big number
   */
  BigNumber addDigit(int digit);

  /**
   * Gets the value of the digit in a position position 0 the digit in the ones place and so on.
   *
   * @param pos the position starts at
   * @return the value of the digit at that position.
   */
  int getDigitAt(int pos);

  /**
   * Returns a copy of the big number which is completely independent of the original.
   *
   * @return the new big number.
   */
  BigNumber copy();

  /**
   * Returns a big number which is the sum of the current big number and the big number give.
   *
   * @param number the big number to add to the current big number
   * @return the new big number which is the sum
   */
  BigNumber add(BigNumber number);

  /**
   * Returns the starting node.
   *
   * @return the startgin node for the big number
   */
  LittleEndianN getStartingNode();

  /**
   * Compares 2 big numbers.
   *
   * @param compare the number to compare to
   * @return 1 if the number to compare to is larger 0 if equal -1 if less than.
   */
  int compareTo(BigNumber compare);
}