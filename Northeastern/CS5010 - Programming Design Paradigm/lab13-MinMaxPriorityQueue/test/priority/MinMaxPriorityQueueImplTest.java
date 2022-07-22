package priority;

import org.junit.Before;
import org.junit.Test;


import static org.junit.Assert.assertEquals;

/**
 * Test for minMaxPriorityQueue.
 */
public class MinMaxPriorityQueueImplTest {
  private MinMaxPriorityQueue<String> testQueue;


  @Before
  public void setUp() {
    testQueue = new MinMaxPriorityQueueImpl<>();
  }

  @Test
  public void testNullGetMin() {
    assertEquals(null, testQueue.minPriorityItem()); 
  }

  @Test
  public void testNullGetMax() {
    assertEquals(null, testQueue.maxPriorityItem());
  }

  @Test
  public void testSameMinPriority() {
    testQueue.add("First", 1);
    testQueue.add("Second", 1);

    assertEquals("First", testQueue.minPriorityItem());
    assertEquals("Second", testQueue.minPriorityItem());
  }

  @Test
  public void testSameMaxPriority() {
    testQueue.add("First", 1);
    testQueue.add("Second", 1);

    assertEquals("First", testQueue.maxPriorityItem());
    assertEquals("Second", testQueue.maxPriorityItem());
  }

  @Test
  public void testMaxPriority() {
    testQueue.add("First", 1);
    testQueue.add("Second", 12);

    assertEquals("Second", testQueue.maxPriorityItem());
  }

  @Test
  public void testMinPriority() {
    testQueue.add("First", 1);
    testQueue.add("Second", 12);

    assertEquals("First", testQueue.minPriorityItem());
  }

}