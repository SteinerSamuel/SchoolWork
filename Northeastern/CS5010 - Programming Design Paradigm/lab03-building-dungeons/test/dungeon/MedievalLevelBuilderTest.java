package dungeon;


import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * A tester class for MedievalLeveBuilder.
 */
public class MedievalLevelBuilderTest {
  MedievalLevelBuilder testBuilder;

  @Before
  public void setUp() throws Exception {
    testBuilder = new MedievalLevelBuilder(1, 1, 1, 1);
  }

  @Test
  public void testConstructor() {
    try {
      testBuilder = new MedievalLevelBuilder(-1, 0, 0, 0);
      fail("IllegalArgumentException should be thrown for a negative level.");
    } catch (IllegalArgumentException e1) {
      try {
        testBuilder = new MedievalLevelBuilder(0, -1, 0, 0);
        fail("IllegalArgumentException should be thrown for a negative room.");
      } catch (IllegalArgumentException e2) {
        try {
          testBuilder = new MedievalLevelBuilder(0, 0, -1, 0);
          fail("IllegalArgumentException should be thrown for a negative monster.");
        } catch (IllegalArgumentException e3) {
          try {
            testBuilder = new MedievalLevelBuilder(0, 0, 0, -1);
            fail("IllegalArgumentException should be thrown for a negative level.");
          } catch (IllegalArgumentException e4) {
            //DO Nothing this is the means we have all the intended behavior.
          }
        }
      }
    }
  }

  @Test
  public void addRoom() {
    testBuilder = new MedievalLevelBuilder(0, 1, 0, 0);
    testBuilder.addRoom("Test Room");
    try {
      testBuilder.addRoom("Test Room 2");
      fail("The builder should only allow one room to be added and throw an IllegalStateException");
    } catch (IllegalStateException e) {
      // Do nothing
    }
    assertEquals("Level 0 contains 1 rooms:\n\nRoom 0 -- Test Room\n"
            + "Monsters:\n\tNone\n"
            + "Treasures:\n\tNone\n", testBuilder.build().toString());
  }

  @Test
  public void addGoblins() {
    testBuilder = new MedievalLevelBuilder(0, 1, 1, 0);
    testBuilder.addRoom("Test Room");
    try {
      testBuilder.addGoblins(0, 2);
      fail();
    } catch (IllegalStateException e) {
      testBuilder.addGoblins(0, 1);
      assertEquals("Level 0 contains 1 rooms:\n\nRoom 0 -- Test Room\n"
              + "Monsters:\n\tgoblin (hitpoints = 7) is a mischievous and very unpleasant, "
              + "vengeful, and greedy creature whose primary purpose is to cause trouble "
              + "to humankind\n"
              + "Treasures:\n\tNone\n", testBuilder.build().toString());
    }

  }

  @Test
  public void addOrc() {
    testBuilder = new MedievalLevelBuilder(0, 1, 1, 0);
    testBuilder.addRoom("Test Room");
    testBuilder.addOrc(0);

    try {
      testBuilder.addOrc(0);
      fail("The builder should of thrown an error you for trying to add more monsters than the "
              + "target");
    } catch (IllegalStateException e) {
      assertEquals("Level 0 contains 1 rooms:\n\n"
                      + "Room 0 -- Test Room\n"
                      + "Monsters:\n\torc (hitpoints = 20) is a brutish, aggressive, malevolent"
                      + " being serving evil\n"
                      + "Treasures:\n\tNone\n",
              testBuilder.build().toString());
    }
  }

  @Test
  public void addOgre() {
    testBuilder = new MedievalLevelBuilder(0, 1, 1, 0);
    testBuilder.addRoom("Test Room");
    testBuilder.addOgre(0);

    try {
      testBuilder.addOgre(0);
      fail("The builder should of thrown an error you for trying to add more monsters than the "
              + "target");
    } catch (IllegalStateException e) {
      assertEquals("Level 0 contains 1 rooms:\n\n"
                      + "Room 0 -- Test Room\n"
                      + "Monsters:\n\togre (hitpoints = 50) is a large, hideous man-like being that"
                      + " likes to eat humans for lunch\n"
                      + "Treasures:\n\tNone\n",
              testBuilder.build().toString());
    }
  }

  @Test
  public void addHuman() {
    testBuilder = new MedievalLevelBuilder(0, 1, 1, 0);
    testBuilder.addRoom("Test Room");
    testBuilder.addHuman(0, "Josh", "just Josh", 10);

    try {
      testBuilder.addHuman(0, "John", "just Jonh", 10);
      fail("Should throw IllegalStateException");
    } catch (IllegalStateException e) {
      assertEquals("Level 0 contains 1 rooms:\n\n"
                      + "Room 0 -- Test Room\n"
                      + "Monsters:\n\tJosh (hitpoints = 10) is a just Josh\n"
                      + "Treasures:\n\tNone\n",
              testBuilder.build().toString());
    }
  }

  @Test
  public void addPotion() {
    testBuilder = new MedievalLevelBuilder(0, 1, 0, 1);
    testBuilder.addRoom("Test Room");
    testBuilder.addPotion(0);

    try {
      testBuilder.addPotion(0);
      fail("The builder should throw an error for adding a treasure past the target.");
    } catch (IllegalStateException e) {
      //Do nothing
    }
  }

  @Test
  public void addGold() {
    testBuilder = new MedievalLevelBuilder(0, 1, 0, 1);
    testBuilder.addRoom("Test Room");
    testBuilder.addGold(0, 10);

    try {
      testBuilder.addGold(0, 10);
      fail("The builder should throw an error for adding a treasure past the target.");
    } catch (IllegalStateException e) {
      //Do nothing
    }
  }

  @Test
  public void addWeapon() {
    testBuilder = new MedievalLevelBuilder(0, 1, 0, 1);
    testBuilder.addRoom("Test Room");
    testBuilder.addWeapon(0, "Sword");

    try {
      testBuilder.addWeapon(0, "Sword");
      fail("The builder should throw an error for adding a treasure past the target.");
    } catch (IllegalStateException e) {
      //Do nothing
    }
  }

  @Test
  public void addSpecial() {
    testBuilder = new MedievalLevelBuilder(0, 1, 0, 1);
    testBuilder.addRoom("Test Room");
    testBuilder.addSpecial(0, "Golden Sword of Magic", 1000);

    try {
      testBuilder.addSpecial(0, "The Sword of Errors", 0);
      fail("The builder should throw an error for adding a treasure past the target.");
    } catch (IllegalStateException e) {
      //Do nothing
    }
  }

  @Test
  public void build() {
    try {
      testBuilder.build();
      fail("Builder should fail here because it should be throwing an exception");
    } catch (IllegalStateException e) {
      testBuilder.addRoom("Test Room");
      try {
        testBuilder.build();
        fail("The build should throw an error here.");
      } catch (IllegalStateException e1) {
        testBuilder.addOrc(0);
        try {
          testBuilder.build();
          fail("The builder should throw an error.");
        } catch (IllegalStateException e2) {
          testBuilder.addGold(0, 100);
          assertEquals("Level 1 contains 1 rooms:\n\n"
                  + "Room 0 -- Test Room\n"
                  + "Monsters:\n\t"
                  + "orc (hitpoints = 20) is a brutish, aggressive, malevolent being serving evil\n"
                  + "Treasures:\n\t"
                  + "pieces of gold (value = 100)\n", testBuilder.build().toString());
        }
      }
    }
  }
}