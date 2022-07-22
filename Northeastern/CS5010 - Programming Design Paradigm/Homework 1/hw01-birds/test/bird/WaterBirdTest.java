package bird;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Water bird test.
 */
public class WaterBirdTest {
  WaterBird duck;

  @Before
  public void setUp() throws Exception {
    duck = new Duck("Howard");
  }

  @Test
  public void getBodyOfWater() {
    assertEquals("Lake", duck.getBodyOfWater());
  }
}