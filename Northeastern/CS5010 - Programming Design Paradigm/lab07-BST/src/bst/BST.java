package bst;

/**
 * Interface for a binary search tree.
 *
 * @param <T> the type of objects the binary search tree holds
 */
public interface BST<T> {
  /**
   * Inserts an object in the tree. If the object is already present in the tree, does nothing.
   *
   * @param object the object to add the tree.
   */
  void add(T object);

  /**
   * Returns the size of the tree (i.e. the number of elements in this tree).
   *
   * @return the size of the tree
   */
  int size();

  /**
   * Returns whether a object is present in the tree.
   *
   * @param obj the object to find.
   * @return true if the object exists in the tree, false if it does not.
   */
  boolean present(T obj);

  /**
   * Returns the smallest object (defined by the ordering) in the tree, and null if the tree is
   * empty.
   *
   * @return the smallest object in the tree or null.
   */
  T minimum();

  /**
   * Returns the largest object (defined by the ordering) in the tree, and null if the tree is
   * empty.
   *
   * @return the largest object in the tree or null.
   */
  T maximum();

  /**
   * Returns the objects in the tree in a preorder manner which is described as follows: process
   * node, traverse left, traverse right.
   *
   * @return a string which represents tree traversal in a PreOrder method.
   */
  String preOrder();

  /**
   * Returns the objects in the tree in a InOrder manner which is described as follows: process
   * traverse left, node, traverse right.
   *
   * @return a string which represents tree traversal in a InOrder method.
   */
  String inOrder();

  /**
   * Returns the objects in the tree in a PreOrder manner which is described as follows: process
   * traverse left, traverse right, node.
   *
   * @return a string which represents tree traversal in a PostOrder method.
   */
  String postOrder();

  /**
   * Calculates the height of the tree.
   *
   * @return the height of the tree
   */
  int height();
}
