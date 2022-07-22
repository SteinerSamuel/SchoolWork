package assignments;

import java.util.Comparator;

/**
 * A comparator which compares assignments by their deadline.
 */
public class DeadlineComparator implements Comparator {
  @Override
  public int compare(Object o1, Object o2) {
    Assignment a1 = (Assignment) o1;
    Assignment a2 = (Assignment) o2;

    if (a1.getEndDate().compareTo(a2.getEndDate()) == 0) {
      return a1.getDescription().compareTo(a2.getDescription());
    } else {
      return (a1.getEndDate().compareTo(a2.getEndDate()));
    }
  }
}
