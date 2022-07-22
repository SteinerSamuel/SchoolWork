package assignments;

import java.util.ArrayList;
import java.util.List;

/**
 * A list of task that need to be completed.
 */
public class AssignmentList {

  private List<Assignment> tasks = new ArrayList<>();
  private String ordering;

  /**
   * Default constructor.
   */
  public AssignmentList() {
    tasks = new ArrayList<>();
    ordering = "assigned";
  }

  /**
   * Add a task to the task list.
   *
   * @param t the task
   */
  public void add(Assignment t) {
    tasks.add(t);
  }

  /**
   * Schedule the assignments on the list using the scheduling strategy provided.
   *
   * @param ss the scheduling strategy to use.
   */
  public void scheduleAssignments(SchedulingStrategy ss) {
    if (ss == null) {
      throw new IllegalArgumentException("Provide a scheduling strategy!");
    } else {
      ordering = ss.schedule(tasks);
    }
  }

  @Override
  public String toString() {
    StringBuffer sb = new StringBuffer("Ordered by ");
    sb.append(ordering);
    sb.append("\n");
    for (int i = 0; i < tasks.size(); i++) {
      sb.append(i + 1);
      sb.append(" -- ");
      sb.append(tasks.get(i));
      sb.append("\n");
    }
    return sb.toString();
  }
}
