package assignments;

import java.util.List;

/**
 * Scheduling strategy implementation which uses the deadline to schedule.
 */
public class DeadlineSchedulingStrategy implements SchedulingStrategy {
  @Override
  public String schedule(List<Assignment> assignments) {
    if (assignments == null || assignments.isEmpty()) {
      throw new IllegalArgumentException("Please provide a valid list of assignments!");
    }
    assignments.sort(new DeadlineComparator());
    return "deadline";
  }
}
