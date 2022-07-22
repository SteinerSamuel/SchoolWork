package bst;

/**
 * A implementation of a tree node which contains an object.
 *
 * @param <T> the type of object
 */
public class ElementNode<T> implements TreeNode<T> {
  private final Comparable<T> value;
  private TreeNode<T> leftNode;
  private TreeNode<T> rightNode;

  /**
   * A constructor of the node which takes a object and gives the node 2 empty nodes.
   *
   * @param object the object the node holds
   */
  public ElementNode(T object) {
    this.value = (Comparable<T>) object;
    this.leftNode = new EmptyNode<>();
    this.rightNode = new EmptyNode<>();
  }

  @Override
  public TreeNode<T> add(T object) {
    int compareResult = this.value.compareTo(object);
    if (compareResult < 0) {
      this.rightNode = this.rightNode.add(object);
    } else if (compareResult > 0) {
      this.leftNode = this.leftNode.add(object);
    } else {
      return this;
    }

    return this;
  }

  @Override
  public boolean present(T obj) {
    int compareResult = value.compareTo(obj);
    if (compareResult > 0) {
      return this.leftNode.present(obj);
    } else if (compareResult < 0) {
      return this.rightNode.present(obj);
    } else {
      return true;
    }
  }

  @Override
  public int size() {
    int leftSize = this.leftNode.size();
    int rightSize = this.rightNode.size();

    return leftSize + rightSize + 1;
  }

  @Override
  public T maximum() {
    if (rightNode.isEmpty()) {
      return (T) this.value;
    } else {
      return rightNode.maximum();
    }
  }

  @Override
  public T minimum() {
    if (leftNode.isEmpty()) {
      return (T) this.value;
    } else {
      return leftNode.maximum();
    }
  }

  /**
   * Return a string of the tree represented in a InOrder method.
   *
   * @return String representation of the tree in order.
   */
  @Override
  public String toString() {
    String leftString;
    String rightString;

    leftString = this.leftNode.toString() + ((this.leftNode.isEmpty()) ? "" : " ");
    rightString = ((this.rightNode.isEmpty()) ? "" : " ") + this.rightNode.toString();

    return leftString + this.value.toString() + rightString;
  }


  @Override
  public String postOrder() {
    String leftString;
    String rightString;
    leftString = this.leftNode.postOrder() + ((this.leftNode.isEmpty()) ? "" : " ");
    rightString = this.rightNode.postOrder() + ((this.rightNode.isEmpty()) ? "" : " ");

    return leftString + rightString + this.value.toString();
  }

  @Override
  public String preOrder() {
    String leftString;
    String rightString;
    leftString = ((this.leftNode.isEmpty()) ? "" : " ") + this.leftNode.preOrder();
    rightString = ((this.rightNode.isEmpty()) ? "" : " ") + this.rightNode.preOrder();

    return this.value.toString() + leftString + rightString;
  }


  @Override
  public int height() {
    int maxHeight = Math.max(leftNode.height(), rightNode.height());

    return 1 + maxHeight;
  }

  @Override
  public boolean isEmpty() {
    return false;
  }
}
