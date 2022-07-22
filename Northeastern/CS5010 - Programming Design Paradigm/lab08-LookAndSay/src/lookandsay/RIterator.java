package lookandsay;

import java.util.Iterator;

/**
 * A Iterator which extends the base iterator which allows for reverse travel of the values.
 *
 * @param <T> The type of object stored in the iterator
 */
public interface RIterator<T> extends Iterator<T> {

  /**
   * Returns whether or not there is a previous value.
   *
   * @return true if there is a previous value false if there isn't.
   */
  boolean hasPrevious();

  /**
   * Returns the previous value.
   *
   * @return The previous value
   */
  T prev();
}
