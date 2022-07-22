package character;

import org.junit.Test;
import org.junit.Before;

import gear.Adjective;
import gear.FeetGear;
import gear.FeetNoun;
import gear.HandGear;
import gear.HandNoun;
import gear.HeadGear;
import gear.HeadNoun;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Character tests.
 */
public class CharacterTest {
  private Character testCharacter;

  @Before
  public void setUp() {
    testCharacter = new CharacterAbstract("Richard", 0, 0);
  }

  @Test
  public void getName() {
    assertEquals("Richard", testCharacter.getName());
  }


  @Test
  public void equipGear() {
    testCharacter.equipGear(new HeadGear(Adjective.afraid, HeadNoun.Cap, 3));
    testCharacter.equipGear(new HeadGear(Adjective.afraid, HeadNoun.FullHelmet, 4));
    try {
      testCharacter.equipGear(new HeadGear(Adjective.afraid, HeadNoun.FullHelmet, 4));
      fail("Should throw error,");
    } catch (IllegalStateException e) {
      // DO nothing expected behavior
    }

    for (int i = 0; i < 4; i++) {
      testCharacter.equipGear(new HandGear(Adjective.afraid, HandNoun.SpikedGloves, 1));
    }
    try {
      testCharacter.equipGear(new HandGear(Adjective.afraid, HandNoun.SpikedGloves, 1));
      fail("Should throw error,");
    } catch (IllegalStateException e) {
      // DO nothing expected behavior
    }

    for (int i = 0; i < 4; i++) {
      testCharacter.equipGear(new FeetGear(Adjective.afraid, FeetNoun.CowboyBoots, 1, 1));
    }
    try {
      testCharacter.equipGear(new FeetGear(Adjective.afraid, FeetNoun.CowboyBoots, 1, 1));
      fail("Should throw error,");
    } catch (IllegalStateException e) {
      // DO nothing expected behavior
    }
  }

  @Test
  public void getAttack() {
    assertEquals(0, testCharacter.getAttack());
    testCharacter.equipGear(new HandGear(Adjective.afraid, HandNoun.SpikedGloves, 1));

    assertEquals(1, testCharacter.getAttack());
  }

  @Test
  public void getDefense() {
    assertEquals(0, testCharacter.getDefense());
    testCharacter.equipGear(new HeadGear(Adjective.afraid, HeadNoun.Earbuds, 1));

    assertEquals(1, testCharacter.getDefense());
  }


}