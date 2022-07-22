package bst;

/**
 * A representation of the an empty node in a tree.
 *
 * @param <T> the type of objects stored in the
 */
public class EmptyNode<T> implements TreeNode<T> {

  @Override
  public TreeNode<T> add(T object) {
    return new ElementNode<>(object);
  }

  @Override
  public boolean present(T obj) {
    return false;
  }

  @Override
  public T maximum() {
    return null;
  }

  @Override
  public T minimum() {
    return null;
  }

  @Override
  public int size() {
    return 0;
  }

  /**
   * Returns an empty string, since node is empty.
   * @return empty string
   */
  @Override
  public String toString() {
    return "";
  }

  @Override
  public String preOrder() {
    return "";
  }

  @Override
  public String postOrder() {
    return "";
  }

  @Override
  public int height() {
    return 0;
  }

  @Override
  public boolean isEmpty() {
    return true;
  }
}
