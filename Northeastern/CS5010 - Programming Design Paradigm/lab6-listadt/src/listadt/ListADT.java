package listadt;

import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * This interface represents a generic list. It is a generalization of the BookListADT interface.
 *
 * <p>We represent the type of data that this will work with a generic parameter T. This is a
 * placeholder for the actual data type.
 *
 * @param <T> the type of element in thee list
 */
public interface ListADT<T> {
  /**
   * Add an object to the front of this list.
   *
   * @param b the object to be added to the front of this list
   */
  void addFront(T b);

  /**
   * Add an object to the back of this list (so it is the last object in the list.
   *
   * @param b the object to be added to teh back of this list
   */
  void addBack(T b);

  /**
   * Add an object to this list so that it occupies the provided index. Index begins with 0
   *
   * @param index the index to be occupied by this object, beginning at 0
   * @param b     the object to be added to the list
   */
  void add(int index, T b);

  /**
   * Return the number of objects currently in this list.
   *
   * @return the size of the list
   */
  int getSize();

  /**
   * Remove the first instance of this object from this list.
   *
   * @param b the object to be removed
   */
  void remove(T b);

  /**
   * Get the (index)th object in this list.
   *
   * @param index the index of the object to be returned
   * @return the object at the given index
   * @throws IllegalArgumentException if an invalid index is passed
   */
  T get(int index) throws IllegalArgumentException;

  /**
   * A general purpose map higher order function on this list, that returns the corresponding list
   * of type R.
   *
   * @param converter the function that converts T into R
   * @param <R>       the type of data in the resulting list
   * @return the resulting list that is identical in structure to this list, but has data of type R
   */
  <R> ListADT<R> map(Function<T, R> converter);

  /**
   * A general purpose filter higher order function on this list that returns the a corresponding
   * list of which is filtered.
   *
   * @param predicate the predicate to filter the list on
   * @return the resulting list that is filtered by the predicate
   */
  ListADT<T> filter(Predicate predicate);

  /**
   * A general purpose fold higher order function which returns a single value of type T using an
   * identity and an accumulator to determine its value.
   *
   * @param identity    the starting value for the fold
   * @param accumulator the method of converting each element in the list into a value
   * @return the final value of the fold in data type t
   */
  T fold(T identity, BinaryOperator<T> accumulator);

  /**
   * Test to see if an element exists in the list.
   *
   * @param element the element to find
   * @return True if the element exists; false if the element does not exist
   */
  boolean anyMatch(T element);
}
