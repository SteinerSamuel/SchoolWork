package bignumber;

/**
 * Linked list node interface for little-endian representation of big number.
 */
public interface LittleEndianN {
  /**
   * Calculates the number of nodes in the linked list.
   *
   * @return 1 + rest if not empty node, 0 if empty node
   */
  int getLength();

  /**
   * Gets the value of the node.
   *
   * @return the value of the node.
   */
  int getValue();

  /**
   * Recursively finds the value at the position.
   *
   * @param pos the position which determines when to terminate
   * @return the value of the position given.
   */
  int valueAt(int pos);

  /**
   * Preforms a left bit shift, this is the same as multiplying by ten.
   *
   * @return The new starting node of the linked list.
   */
  LittleEndianN shiftLeft();

  /**
   * Preforms a right bit shift this is the same as integer division by ten.
   *
   * @return the new starting node of the linked list.
   */
  LittleEndianN shiftRight();

  /**
   * Adds a value to the current node.
   *
   * @param value The value to be added.
   * @return the starting node of the new linked linked list
   */
  LittleEndianN add(int value);

  /**
   * Adds a node to the current node and will do carry over math as well.
   *
   * @param node      the node to be added
   * @param carryover the carry over value if any
   * @return the new node with the updated value
   */
  LittleEndianN addNode(LittleEndianN node, int carryover);

  /**
   * Gets the next node in the number.
   *
   * @return the next node in the number
   */
  LittleEndianN getRest();
}
