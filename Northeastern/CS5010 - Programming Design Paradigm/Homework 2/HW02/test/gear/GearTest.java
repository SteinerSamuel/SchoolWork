package gear;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Class for testing gear.
 */
public class GearTest {
  Gear g1;
  Gear g2;
  Gear g3;

  @Before
  public void setUp() {
    g1 = new FeetGear(Adjective.afraid, FeetNoun.Boots, 2, 3);
    g2 = new HeadGear(Adjective.clever, HeadNoun.Cap, 3);
    g3 = new HandGear(Adjective.hallowed, HandNoun.BrassKnuckles, 300);
  }

  @Test
  public void testFeetGearConstructor() {
    try {
      g1 = new FeetGear(null, FeetNoun.Boots, 30, 0);
      fail("Should throw an exception!");
    } catch (IllegalArgumentException e) {
      try {
        g1 = new FeetGear(Adjective.afraid, null, 10, 0);
        fail("Should throw an exception!");
      } catch (IllegalArgumentException e1) {
        //  Do nothing expected behavior.
      }
    }
  }

  @Test
  public void testHeadGearConstructor() {
    try {
      g1 = new HeadGear(null, HeadNoun.MidHelmet, 30);
      fail("Should throw an exception!");
    } catch (IllegalArgumentException e) {
      try {
        g1 = new HeadGear(Adjective.afraid, null, 10);
        fail("Should throw an exception!");
      } catch (IllegalArgumentException e1) {
        //  Do nothing expected behavior.
      }
    }
  }

  @Test
  public void testHandGearConstructor() {
    try {
      g1 = new HandGear(null, HandNoun.Sword, 30);
      fail("Should throw an exception!");
    } catch (IllegalArgumentException e) {
      try {
        g1 = new HandGear(Adjective.afraid, null, 10);
        fail("Should throw an exception!");
      } catch (IllegalArgumentException e1) {
        //  Do nothing expected behavior.
      }
    }
  }

  @Test
  public void getName() {
    assertEquals("hallowed BrassKnuckles", g3.getName());
    assertEquals("afraid Boots", g1.getName());
    assertEquals("clever Cap", g2.getName());
  }

  @Test
  public void getAttack() {
    assertEquals(3, g1.getAttack());
    assertEquals(0, g2.getAttack());
    assertEquals(300, g3.getAttack());
  }

  @Test
  public void getDefense() {
    assertEquals(2, g1.getDefense());
    assertEquals(3, g2.getDefense());
    assertEquals(0, g3.getDefense());
  }

  @Test
  public void getCombined() {
    Gear test = new HandGear(Adjective.maddening, HandNoun.SpikedGloves, 20);
    test = g3.combine(test);
    assertFalse(g1.getCombined());
    assertFalse(g2.getCombined());
    assertFalse(g3.getCombined());
    assertTrue(test.getCombined());
  }

  @Test
  public void combine() {
    Gear test = new HandGear(Adjective.maddening, HandNoun.SpikedGloves, 20);
    test = g3.combine(test);
    assertTrue(test.getCombined());
  }
}