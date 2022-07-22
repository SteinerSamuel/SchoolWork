package assignments;

import org.junit.Before;
import org.junit.Test;


import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Test cases for the schedulers.
 */
public class AssignmentListTest {
  AssignmentList testlist1;
  AssignmentList testlist2;
  SchedulingStrategy ss;

  @Before
  public void setUp() {
    testlist1 = new AssignmentList();
    testlist2 = new AssignmentList();
    testlist2.add(new Assignment("Test 1"));
    Assignment a2 = new Assignment("Test 4");
    a2.setDeadline(4, 20, 2021);
    a2.setStart(3, 24, 2021);
    Assignment a3 = new Assignment("Test 3");
    a3.setDeadline(1, 20, 2022);
    a3.setStart(12, 24, 2021);
    testlist2.add(a2);
    testlist2.add(a3);
    testlist2.add(new Assignment("Test 2"));
  }

  @Test
  public void testNoSchedulerError() {
    ss = null;
    try {
      testlist1.scheduleAssignments(ss);
      fail("Should throw exception.");
    } catch (IllegalArgumentException e) {
      // Do nothing
    }
  }

  @Test
  public void testAlphabeticalSchedulerString() {
    ss = new AlphabeticalSchedulingStrategy();
    testlist2.scheduleAssignments(ss);

    String expected = "Ordered by alphabetical\n"
            + "1 -- Test 1, starting 2020-11-15, ending 2020-11-15\n"
            + "2 -- Test 2, starting 2020-11-15, ending 2020-11-15\n"
            + "3 -- Test 3, starting 2021-12-24, ending 2022-01-20\n"
            + "4 -- Test 4, starting 2021-03-24, ending 2021-04-20\n";


    assertEquals(expected, testlist2.toString());
  }

  @Test
  public void testAssignedSchedulerString() {
    ss = new AssignedSchedulingStrategy();
    testlist2.scheduleAssignments(ss);

    String expected = "Ordered by assigned\n"
            + "1 -- Test 1, starting 2020-11-15, ending 2020-11-15\n"
            + "2 -- Test 4, starting 2021-03-24, ending 2021-04-20\n"
            + "3 -- Test 3, starting 2021-12-24, ending 2022-01-20\n"
            + "4 -- Test 2, starting 2020-11-15, ending 2020-11-15\n";

    assertEquals(expected, testlist2.toString());
  }

  @Test
  public void testDeadlineSchedulerString() {
    ss = new DeadlineSchedulingStrategy();
    testlist2.scheduleAssignments(ss);

    String expected = "Ordered by deadline\n"
            + "1 -- Test 1, starting 2020-11-15, ending 2020-11-15\n"
            + "2 -- Test 2, starting 2020-11-15, ending 2020-11-15\n"
            + "3 -- Test 4, starting 2021-03-24, ending 2021-04-20\n"
            + "4 -- Test 3, starting 2021-12-24, ending 2022-01-20\n";

    assertEquals(expected, testlist2.toString());
  }

  @Test
  public void testDifficultySchedulerString() {
    ss = new DifficultySchedulingStrategy();
    testlist2.scheduleAssignments(ss);

    String expected = "Ordered by difficulty\n"
            + "1 -- Test 1, starting 2020-11-15, ending 2020-11-15\n"
            + "2 -- Test 2, starting 2020-11-15, ending 2020-11-15\n"
            + "3 -- Test 3, starting 2021-12-24, ending 2022-01-20\n"
            + "4 -- Test 4, starting 2021-03-24, ending 2021-04-20\n";

    assertEquals(expected, testlist2.toString());
  }


  @Test
  public void testAlphabeticalSchedulerNoList() {
    ss = new AlphabeticalSchedulingStrategy();

    try {
      testlist1.scheduleAssignments(ss);
      fail("Should throw exception;");
    } catch (IllegalArgumentException e) {
      //Do nothing
    }
  }

  @Test
  public void testAssignedSchedulerNoList() {
    ss = new AssignedSchedulingStrategy();

    try {
      testlist1.scheduleAssignments(ss);
      fail("Should throw exception;");
    } catch (IllegalArgumentException e) {
      //Do nothing
    }
  }

  @Test
  public void testDeadlineSchedulerNoList() {
    ss = new DeadlineSchedulingStrategy();

    try {
      testlist1.scheduleAssignments(ss);
      fail("Should throw exception;");
    } catch (IllegalArgumentException e) {
      //Do nothing
    }
  }

  @Test
  public void testDifficultylineSchedulerNoList() {
    ss = new DifficultySchedulingStrategy();

    try {
      testlist1.scheduleAssignments(ss);
      fail("Should throw exception;");
    } catch (IllegalArgumentException e) {
      //Do nothing
    }
  }
}