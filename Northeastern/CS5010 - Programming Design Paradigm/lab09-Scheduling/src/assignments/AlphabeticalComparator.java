package assignments;

import java.util.Comparator;

/**
 * Comparator used which compares Assignments by their description alphabetically.
 */
public class AlphabeticalComparator implements Comparator {

  @Override
  public int compare(Object o1, Object o2) {
    Assignment a1 = (Assignment) o1;
    Assignment a2 = (Assignment) o2;

    return a1.getDescription().compareTo(a2.getDescription());
  }
}
