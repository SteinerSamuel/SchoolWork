package bird;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Talking bird test.
 */
public class TalkingBirdTest {
  TalkingBird rosie;

  @Before
  public void setUp() {
    rosie = new RoseRingParakeet("Rosie", 30, "Kisses!");
  }

  @Test
  public void getVocabSize() {
    assertEquals(30, rosie.getVocabSize());
  }

  @Test
  public void getFavoritePhrase() {
    assertEquals("Kisses!", rosie.getFavoritePhrase());
  }
}