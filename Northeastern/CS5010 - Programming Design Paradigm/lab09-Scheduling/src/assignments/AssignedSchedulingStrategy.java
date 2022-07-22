package assignments;

import java.util.List;

/**
 * Scheduling Strategy implementation which schedules based on when the assignment was assigned.
 */
public class AssignedSchedulingStrategy implements SchedulingStrategy {
  @Override
  public String schedule(List<Assignment> assignments) {
    if (assignments == null || assignments.isEmpty()) {
      throw new IllegalArgumentException("Please provide a valid list of assignments!");
    }
    assignments.sort(new AssignedComparator());
    return "assigned";
  }
}
