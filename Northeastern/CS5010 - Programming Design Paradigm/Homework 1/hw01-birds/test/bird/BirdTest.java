package bird;

import org.junit.Test;
import org.junit.Before;

import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Bird test
 */
public class BirdTest {
  private Bird hawk;
  private Bird kiwi;
  private Bird greatAuk;
  private Bird owl;
  private Bird roseRingParakeet;

  @Before
  public void setUp() {
    hawk = new Hawk("Henry");
    kiwi = new Kiwi("Kert");
    greatAuk = new GreatAuk("George");
    owl = new Owl("Otis", false, 2, new ArrayList<Food>(
            Arrays.asList(Food.SMALL_MAMMALS, Food.OTHER_BIRDS)));
    roseRingParakeet = new RoseRingParakeet("Rosie", 20, "Hello!");
  }

  @Test
  public void getExtinct() {
    assertFalse(hawk.getExtinct());
    assertTrue(kiwi.getExtinct());
    assertTrue(greatAuk.getExtinct());
    assertFalse(owl.getExtinct());
    assertFalse(roseRingParakeet.getExtinct());
  }

  @Test
  public void getBirdName() {
    assertEquals("Henry", hawk.getBirdName());
    assertEquals("Kert", kiwi.getBirdName());
    assertEquals("George", greatAuk.getBirdName());
    assertEquals("Otis", owl.getBirdName());
    assertEquals("Rosie", roseRingParakeet.getBirdName());
  }

  @Test
  public void getNumOfWings() {
    assertEquals(2, hawk.getNumOfWings());
    assertEquals(0, kiwi.getNumOfWings());
    assertEquals(2, greatAuk.getNumOfWings());
    assertEquals(2, owl.getNumOfWings());
    assertEquals(2, roseRingParakeet.getNumOfWings());
  }

  @Test
  public void getDietPreference() {
    assertEquals(new ArrayList<Food>(Arrays.asList(Food.SMALL_MAMMALS, Food.OTHER_BIRDS)),
            owl.getDietPreference());
    assertEquals(new ArrayList<>(Arrays.asList(Food.BUDS, Food.VEGETATION)),
            kiwi.getDietPreference());

  }

  @Test
  public void getCharacteristics() {
    assertEquals(new ArrayList<>(Arrays.asList("Long beak", "Flexible beak", "Flightless")),
            kiwi.getCharacteristics());
  }

  @Test
  public void testToString() {
    assertEquals("Kert is a Kiwi: Kiwis are  extinct have 0 wings and eat buds, "
            + "vegetation. They have the following characteristics: "
            + "Long beak, Flexible beak, Flightless. \n", kiwi.toString());
  }
}