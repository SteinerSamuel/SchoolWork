package assignments;


import java.util.List;

/**
 * Scheduling strategy implementation which uses difficulty to schedule tasks.
 */
public class DifficultySchedulingStrategy implements SchedulingStrategy {
  @Override
  public String schedule(List<Assignment> assignments) {
    if (assignments == null || assignments.isEmpty()) {
      throw new IllegalArgumentException("Please provide a valid list of assignments!");
    }
    assignments.sort(new DifficultyComparator());
    return "difficulty";
  }
}
