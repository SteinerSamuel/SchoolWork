package assignments;

import java.util.List;

/**
 * Scheduling strategy implementation which schedules by alphabetical order.
 */
public class AlphabeticalSchedulingStrategy implements SchedulingStrategy {
  @Override
  public String schedule(List<Assignment> assignments) {
    if (assignments == null || assignments.isEmpty()) {
      throw new IllegalArgumentException("Please provide a valid list of assignments!");
    }
    assignments.sort(new AlphabeticalComparator());
    return "alphabetical";
  }
}