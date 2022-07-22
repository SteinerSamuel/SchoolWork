package conservatory;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import javax.naming.SizeLimitExceededException;

import aviary.Aviary;
import bird.Bird;
import bird.Food;
import bird.Hawk;
import bird.Owl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Junit test for conservatory.
 */
public class ConservatoryTest {
  Conservatory conservatoryTest;
  Bird bird;

  @Before
  public void setUp() throws Exception {
    bird = new Hawk("Henry");
    conservatoryTest = new ConservatoryConcrete();
  }

  // Test get aviary but also tests add aviary.
  @Test
  public void getAviaries() throws SizeLimitExceededException {
    Aviary test1 = conservatoryTest.addAviary("Test Location");

    assertEquals(new ArrayList<Aviary>(Arrays.asList(test1)), conservatoryTest.getAviaries());
  }

  @Test
  public void addBird() throws SizeLimitExceededException {
    Aviary test1 = conservatoryTest.addAviary("Test Location");

    conservatoryTest.addBird(bird, test1);

    assertTrue(test1.hasBird(bird));

  }

  @Test
  public void calcFood() throws SizeLimitExceededException {
    Aviary test1 = conservatoryTest.addAviary("Test Location");
    conservatoryTest.addBird(bird, test1);
    HashMap<Food, Integer> testHash = new HashMap<Food, Integer>();
    testHash.put(Food.OTHER_BIRDS, 1);
    testHash.put(Food.SMALL_MAMMALS, 1);
    testHash.put(Food.FISH, 1);

    assertEquals(testHash.toString(), conservatoryTest.calcFood());
  }

  @Test
  public void findBird() throws SizeLimitExceededException {
    Aviary test1 = conservatoryTest.addAviary("Test");
    conservatoryTest.addBird(bird, test1);

    assertEquals(test1, conservatoryTest.findBird(bird));
  }

  @Test
  public void aviaryDescription() throws SizeLimitExceededException {
    Aviary test1 = conservatoryTest.addAviary("Test");
    conservatoryTest.addBird(bird, test1);

    assertEquals("This is the aviary located at Test. The following birds are "
            + "in this aviary:\n"
            + "Henry is a Hawk: Hawks are not extinct have 2 wings and eat small mammals, fish, "
            + "other birds. They have the following characteristics:"
            + " Large, Sharp beak, Visible nostrils. \n" +
            "\n", conservatoryTest.getDirectory());
  }

  @Test
  public void testIndex() throws SizeLimitExceededException {
    Aviary test1 = conservatoryTest.addAviary("Test");
    conservatoryTest.addBird(bird, test1);

    assertEquals("Henry is in the aviary located at Test", conservatoryTest.getIndex());
  }

}