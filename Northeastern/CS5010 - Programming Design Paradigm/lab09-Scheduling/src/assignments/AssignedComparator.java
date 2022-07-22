package assignments;

import java.util.Comparator;

/**
 *  A comparator which compares when the assignment was assigned.
 */
public class AssignedComparator implements Comparator {
  @Override
  public int compare(Object o1, Object o2) {
    Assignment a1 = (Assignment) o1;
    Assignment a2 = (Assignment) o2;

    if (a1.getNumber() == a2.getNumber()) {
      return a1.getDescription().compareTo(a2.getDescription());
    } else if (a1.getNumber() < a2.getNumber()) {
      return -1;
    } else {
      return 1;
    }
  }
}
