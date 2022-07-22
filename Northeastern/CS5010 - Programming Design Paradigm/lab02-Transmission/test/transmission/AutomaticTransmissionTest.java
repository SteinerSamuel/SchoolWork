package transmission;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.Random;


/**
 * Test for AutomaticTransmission class.
 */
public class AutomaticTransmissionTest {
  AutomaticTransmission testTransmission1;
  AutomaticTransmission testTransmission2;

  @Before
  public void setUp() {
    testTransmission1 = new AutomaticTransmission(10, 20, 30, 40, 50);
  }

  /**
   * Test the constructor to test if the constructor throws an error when parameters do not follow
   * the rules which are set val1 < val2 < val3 < val4 < val5.
   */
  @Test
  public void testIllegalArgument() {
    int val1;
    int val2;
    int val3;
    int val4;
    int val5;
    int largeVal = 300;
    Random rand = new Random();
    for (int i = 0; i < 5; i++) {
      val1 = rand.nextInt(25);
      val2 = rand.nextInt(25) + 25;
      val3 = rand.nextInt(25) + 50;
      val4 = rand.nextInt(25) + 75;
      val5 = rand.nextInt(25) + 100;
      try {
        testTransmission2 = new AutomaticTransmission(val1, val2, val3, val4, val5);
      } catch (IllegalArgumentException e) {
        fail("Illegal Argument exception should not be thrown;");
      }
    }
    for (int i = 0; i < 4; i++) {
      val1 = rand.nextInt(25);
      val2 = rand.nextInt(25) + 25;
      val3 = rand.nextInt(25) + 50;
      val4 = rand.nextInt(25) + 75;
      val5 = rand.nextInt(25) + 100;
      switch (i) {
        case 0:
          val1 = largeVal;
          break;
        case 1:
          val2 = largeVal;
          break;
        case 2:
          val3 = largeVal;
          break;
        case 3:
          val4 = largeVal;
          break;
        default:
          val5 = largeVal;
          break;
      }
      try {
        testTransmission2 = new AutomaticTransmission(val1, val2, val3, val4, val5);
        fail(String.format("Test failed breakPoint%d should of thrown a failure.", i + 1));
      } catch (IllegalArgumentException e) {
        // Do nothing this is the intended functionality.
      }
    }
  }

  @Test
  public void testGetSpeed() {
    assertEquals(0, testTransmission1.getSpeed());
    for (int i = 1; i <= 5; i++) {
      testTransmission1 = (AutomaticTransmission) testTransmission1.increaseSpeed();
      assertEquals((i) * 2, testTransmission1.getSpeed());
    }
  }

  @Test
  public void testIncreaseSpeed() {
    for (int i = 1; i <= 5; i++) {
      testTransmission1 = (AutomaticTransmission) testTransmission1.increaseSpeed();
      assertEquals(i * 2, testTransmission1.getSpeed());
    }
  }

  @Test
  public void testGetGear() {
    assertEquals(0, testTransmission1.getGear());
    // increase the speed 25 times this will makes sure testTransmission will go through every gear.
    for (int i = 1; i < 25; i++) {
      testTransmission1 = (AutomaticTransmission) testTransmission1.increaseSpeed();
      // if its the first time or any time we pass a threshold (every 5 times) we should see the
      // next gear
      if (i % 5 == 0 || i == 1) {
        // we can calculate what gear we should be in by dividing i by 5 and then adding 1.
        assertEquals((i / 5) + 1, testTransmission1.getGear());
      }
    }
  }


  @Test
  public void testDecreaseSpeed() {
    try {
      testTransmission1.decreaseSpeed();
      fail("An illegal state error should have been thrown");
    } catch (IllegalStateException e) {
      // Do nothing this is the expected behavior
    }
    // create a transmission which already has a speed value in this case 40
    testTransmission2 = new AutomaticTransmission(10, 20, 30, 40, 50, 40);

    for (int i = 1; i <= 5; i++) {
      testTransmission2 = (AutomaticTransmission) testTransmission2.decreaseSpeed();
      assertEquals(40 - (i * 2), testTransmission2.getSpeed());
    }
  }

  @Test
  public void testToString() {
    assertEquals("Transmission (speed = 0, gear = 0)", testTransmission1.toString());
  }
}