package listadt;

import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * This is the implementation of a generic list. Specifically it implements the listadt.ListADT
 * interface
 *
 * @param <T> the type of element in this list
 */
public class ListADTImpl<T> implements ListADT<T> {
  private GenericListADTNode<T> head;

  /**
   * Default constructor.
   */
  public ListADTImpl() {
    head = new GenericEmptyNode<>();
  }

  /**
   * A private constructor that is used internally (see map).
   *
   * @param head the head of this list
   */
  private ListADTImpl(GenericListADTNode<T> head) {
    this.head = head;
  }

  @Override
  public void addFront(T b) {
    head = head.addFront(b);
  }

  @Override
  public void addBack(T b) {
    head = head.addBack(b);
  }

  @Override
  public void add(int index, T b) {
    head = head.add(index, b);
  }

  @Override
  public int getSize() {
    return head.count();
  }

  @Override
  public void remove(T b) {
    head = head.remove(b);
  }

  @Override
  public T get(int index) throws IllegalArgumentException {
    if ((index >= 0) && (index < getSize())) {
      return head.get(index);
    } else {
      throw new IllegalArgumentException("Invalid index");
    }
  }

  @Override
  public <R> ListADT<R> map(Function<T, R> converter) {
    return new ListADTImpl<>(head.map(converter));
  }

  @Override
  public ListADT<T> filter(Predicate predicate) {
    return new ListADTImpl<>(head.filter(predicate));
  }

  @Override
  public T fold(T identity, BinaryOperator<T> accumulator) {
    return head.fold(identity, accumulator);
  }

  @Override
  public String toString() {
    return "(" + head.toString() + ")";
  }

  @Override
  public boolean anyMatch(T element) {
    return head.match(element);
  }

  // protected methods for implementing swap and reverse

  /**
   * Gets the head node of the list.
   *
   * @return the node which is at the head of the list
   */
  protected GenericListADTNode<T> getHead() {
    return head;
  }

  /**
   * Set the head to the new head passed to this method.
   *1
   * @param newHead the new head to set the head of this list too.
   */
  protected void setHead(GenericListADTNode<T> newHead) {
    this.head = newHead;
  }


}
