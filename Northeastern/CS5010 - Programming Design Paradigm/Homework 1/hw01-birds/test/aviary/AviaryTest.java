package aviary;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;

import javax.naming.SizeLimitExceededException;

import bird.Bird;
import bird.Emu;
import bird.GreatAuk;
import bird.Hawk;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Aviary Test.
 */
public class AviaryTest {
  Aviary aviaryTest;
  Bird testBird1;
  Bird testBird2;
  Bird testBird3;
  Bird testBird4;
  Bird testBird5;
  Bird testBird6;


  @Before
  public void setUp() {
    aviaryTest = new AviaryConcrete("Lobby");
    testBird1 = new Hawk("Hal");
    testBird2 = new Hawk("Henry");
    testBird3 = new Hawk("Haley");
    testBird4 = new Hawk("Harvey");
    testBird5 = new Hawk("Harold");
    testBird6 = new Hawk("Hillary");
  }

  @Test
  public void getBirds() throws SizeLimitExceededException {
    aviaryTest.addBird(testBird1);
    aviaryTest.addBird(testBird2);
    aviaryTest.addBird(testBird3);
    aviaryTest.addBird(testBird4);

    assertEquals(new ArrayList<Bird>(Arrays.asList(testBird1, testBird2, testBird3, testBird4)),
            aviaryTest.getBirds());
  }

  @Test
  public void getLocation() {
    assertEquals("Lobby", aviaryTest.getLocation());
  }

  @Test
  public void isEmpty() throws SizeLimitExceededException {
    assertTrue(aviaryTest.isEmpty());
    aviaryTest.addBird(testBird1);
    assertFalse(aviaryTest.isEmpty());
  }

  @Test
  public void hasBird() throws SizeLimitExceededException {
    aviaryTest.addBird(testBird1);

    assertTrue(aviaryTest.hasBird(testBird1));
    assertFalse(aviaryTest.hasBird(testBird2));
  }

  @Test
  public void addBird() throws SizeLimitExceededException {
    // add birds till we have 5
    aviaryTest.addBird(testBird1);
    aviaryTest.addBird(testBird2);
    aviaryTest.addBird(testBird3);
    aviaryTest.addBird(testBird4);
    aviaryTest.addBird(testBird5);

    try {
      aviaryTest.addBird(testBird6);
      fail("Should have failed. Cannot add more than 5 birds.");
    } catch (SizeLimitExceededException e) {
      System.out.println(e);
    }
  }
}