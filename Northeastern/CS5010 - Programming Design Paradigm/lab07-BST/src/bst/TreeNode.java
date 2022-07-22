package bst;

/**
 * An interface for a tree node, these nodes represent 1 object in the tree.
 *
 * @param <T> the types of objects
 */
public interface TreeNode<T> {
  /**
   * Adds an object to the tree. If the object already exists in the tree does nothing
   *
   * @param object The object to be added.
   * @return The root of the tree.
   */
  TreeNode<T> add(T object);

  /**
   * Returns whether a object is present in the tree.
   *
   * @param obj the object to find.
   * @return true if the object exists in the tree, false if it does not.
   */
  boolean present(T obj);

  /**
   * Returns the size of the tree (i.e. the number of elements in this tree).
   *
   * @return the size of the tree
   */
  int size();

  /**
   * Returns the largest object (defined by the ordering) in the tree, and null if the tree is
   * empty.
   *
   * @return the largest object in the tree or null.
   */
  T maximum();

  /**
   * Returns the smallest object (defined by the ordering) in the tree, and null if the tree is
   * empty.
   *
   * @return the smallest object in the tree or null.
   */
  T minimum();

  /**
   * Returns the objects in the tree in a preorder manner which is described as follows: process
   * node, traverse left, traverse right.
   *
   * @return a string which represents tree traversal in a PreOrder method.
   */
  String preOrder();

  /**
   * Returns the objects in the tree in a PreOrder manner which is described as follows: process
   * traverse left, traverse right, node.
   *
   * @return a string which represents tree traversal in a PostOrder method.
   */
  String postOrder();

  /**
   * Calculates the height of the tree from the current node.
   *
   * @return the height of the tree
   */
  int height();

  /**
   * Is this node empty.
   *
   * @return true if the node is empty false if not.
   */
  boolean isEmpty();
}
