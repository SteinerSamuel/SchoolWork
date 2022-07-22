package listadt;

import java.util.Arrays;
import java.util.function.Predicate;

/**
 * Utilities for the ListADT.
 */
public final class ListADTUtilities {
  /**
   * Takes and array of elements and returns a ADT with all of the elements.
   *
   * @param elements an array of objects to be turned into an ADT list
   * @param <T>      the type of the objects in the array and the list
   * @return an ADT list which has the objects which appear in the array in the same order as they
   *        appear
   */
  public static <T> ListADT<T> toList(T[] elements) {

    ListADT<T> list = new ListADTImpl<T>();

    for (T element : elements) {
      if (element == null) {
        throw new IllegalArgumentException("Array contains null value(s)");
      }
      list.addBack(element);
    }

    return list;
  }

  /**
   * Adds the elements given to the list given in the order they are given. If a null is present no
   * values will be added to the list.
   *
   * @param list     the list to add to
   * @param elements the elements to add to the list
   * @param <T>      the type the list and elements are
   */
  public static <T> void addAll(ListADT<T> list, T... elements) {
    if (Arrays.asList(elements).contains(null)) {
      throw new IllegalArgumentException("Elements given contains null value(s)");
    }

    for (T element : elements) {
      list.addBack(element);
    }
  }

  /**
   * Calculates the frequency of an element in a list.
   *
   * @param list    The list to calculate the frequency from
   * @param element the element to count the frequency
   * @param <T>     the type of the list and the element
   * @return the count of the elements if the element doesnt exist returns 0
   */
  public static <T> int frequency(ListADT<T> list, T element) {
    ListADT<T> filtered = list.filter(Predicate.isEqual(element));
    return filtered.getSize();
  }

  /**
   * Returns whether the two given lists are disjointed, two lists are disjointed when they do not
   * share any elements. This is regardless of position.
   *
   * @param one the first list to compare
   * @param two the second list to compare
   * @return true if the two lists are disjointed; false if the two lists are not disjointed
   */
  public static boolean disjoint(ListADT<?> one, ListADT<?> two) {
    if (one == null || two == null
            || frequency(one, null) > 0 || frequency(two, null) > 0) {
      throw new IllegalArgumentException("One of the lists contains (a) null value(s)");
    }

    if (one.getSize() > two.getSize()) {
      for (int i = 0; i < two.getSize(); i++) {
        if (!(disjointHelper(one, two, i))) {
          return false;
        }
      }
    } else {
      for (int i = 0; i < one.getSize(); i++) {
        if (!(disjointHelper(two, one, i))) {
          return false;
        }
      }
    }


    return true;
  }

  /**
   * Helper function for disjoint.
   *
   * @param one the first list
   * @param two the second list
   * @param i   the index of the testing object
   * @param <T> the type of list one
   * @param <U> the type of list two
   * @return return false if there the lists are do not contain the match returns true if the lists
   *        do contain the match.
   */
  private static <T, U> boolean disjointHelper(ListADT<T> one, ListADT<U> two, int i) {
    try {
      return !one.anyMatch((T) two.get(i));
    } catch (ClassCastException e) {
      return true;
    }
  }

  /**
   * Returns whether the two given lists are equal. Two lists are equal when they have the same
   * elements in the same position as one another.
   *
   * @param one the first list to compare
   * @param two the second list to compare
   * @return True if the lists are equal; false if the two lists are not equal
   */
  public static boolean equals(ListADT<?> one, ListADT<?> two) {
    if (one == null || two == null ||
            frequency(one, null) > 0 || frequency(two, null) > 0) {
      throw new IllegalArgumentException("One of the lists contains (a) null value(s)");
    }

    if (one.getSize() != two.getSize()) {
      return false;
    }

    for (int i = 0; i < two.getSize(); i++) {
      if (!(one.get(i).equals(two.get(i)))) {
        return false;
      }
    }

    return true;
  }

  /**
   * Swaps 2 elements of the list.
   *
   * @param list the list which to swap the elements in
   * @param i    the first index bound from 0 to the length of the list -1
   * @param j    the second index bound from 0 to the length of the list -1
   */
  public static void swap(ListADT<?> list, int i, int j) {
    if ((i >= 0 && i < list.getSize()) && (j > 0 && j < list.getSize())) {
      if (i != j) {
        Object o = list.get(i);
        Object b = list.get(j);
        Object[] e = new Object[list.getSize()];
        for (int l = 0; l < list.getSize(); l++) {
          if (l == i) {
            e[l] = b;
          } else if (l == j) {
            e[l] = o;
          } else {
            e[l] = list.get(l);
          }
        }
        ((ListADTImpl) list).setHead(((ListADTImpl) toList(e)).getHead());
      }
    } else {
      throw new IndexOutOfBoundsException("One or both of the values provided are out of bounds.");
    }
  }

  /**
   * Reverses the order of a list.
   *
   * @param list the list to reverse the order of the elements
   */
  public static void reverse(ListADT<?> list) {
    int i = 0;
    int j = list.getSize() - 1;

    while (i < j) {
      swap(list, i, j);
      i++;
      j--;
    }
  }

}