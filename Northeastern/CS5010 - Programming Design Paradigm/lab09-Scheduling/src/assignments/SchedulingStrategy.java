package assignments;

import java.util.List;

/**
 * Strategy design model interface for a scheduler.
 */
public interface SchedulingStrategy {
  /**
   * Schedule method which which returns the type of schedule.
   *
   * @param assignments the list of assignments
   * @return the type of scheduling example "alphabetical"
   */
  String schedule(List<Assignment> assignments);
}
