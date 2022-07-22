package assignments;

import java.util.Comparator;

/**
 * Comparator which uses difficulty as a way to compare.
 */
public class DifficultyComparator implements Comparator {
  @Override
  public int compare(Object o1, Object o2) {
    Assignment a1 = (Assignment) o1;
    Assignment a2 = (Assignment) o2;

    if (a1.getDifficulty() == a2.getDifficulty()) {
      return a1.getDescription().compareTo(a2.getDescription());
    } else if (a1.getDifficulty() < a2.getDifficulty()) {
      return -1;
    } else {
      return 1;
    }
  }
}
