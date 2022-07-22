package bst;

/**
 * Implementation of a binary search tree.
 *
 * @param <T> the type of objects stored.
 */
public class BSTImpl<T> implements BST<T> {
  private TreeNode<T> root;

  /**
   * Constructor for a binary search tree, the tree is empty.
   */
  public BSTImpl() {
    this.root = new EmptyNode();
  }

  @Override
  public void add(T object) {
    this.root = root.add(object);
  }

  @Override
  public boolean present(T obj) {
    return root.present(obj);
  }

  @Override
  public int size() {
    return root.size();
  }

  @Override
  public T minimum() {
    return root.minimum();
  }

  @Override
  public T maximum() {
    return root.maximum();
  }

  /**
   * Return a string of the tree represented in a InOrder method.
   *
   * @return String representation of the tree in order.
   */
  @Override
  public String toString() {
    return "[" + root.toString() + "]";
  }

  @Override
  public String inOrder() {
    return this.toString();
  }

  @Override
  public String preOrder() {
    return "[" + root.preOrder() + "]";
  }

  @Override
  public String postOrder() {
    return "[" + root.postOrder() + "]";
  }

  @Override
  public int height() {
    return root.height();
  }
}
