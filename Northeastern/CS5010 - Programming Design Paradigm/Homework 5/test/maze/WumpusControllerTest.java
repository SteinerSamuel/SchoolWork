package maze;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;


/**
 * Test class for hte controller.
 */
public class WumpusControllerTest {
  WumpusController wc;

  @Before
  public void setUp() {
    wc = new WumpusController(5, 5, true, false, 0, 0, 123, 20, 20);
  }

  @Test
  public void getPlayerPos() {
    assertEquals(new Coordinates(0, 0), wc.getPlayerPos());
  }

  @Test
  public void testGetPlayerMoves() {
    ArrayList<CardinalDirections> expected = new ArrayList<>(Arrays.asList(
            CardinalDirections.SOUTH, CardinalDirections.EAST));

    assertEquals(expected, wc.getPlayerMoves());
  }

  @Test
  public void testMovePlayer() {
    assertEquals(new Coordinates(1, 0), wc.movePlayer(CardinalDirections.EAST));
  }

  @Test
  public void testGetAdjacent() {
    ArrayList<Contents> expected = new ArrayList<>(Arrays.asList(
            Contents.NOTHING, Contents.NOTHING
    ));
    assertEquals(expected, wc.getAdjacent());
  }

  @Test
  public void testShoot() {
    assertFalse(wc.shoot(CardinalDirections.EAST, 9));
  }

  @Test
  public void getContent() {
    assertEquals(Contents.NOTHING, wc.getContent());
  }

  @Test
  public void bat() {
    boolean moved = false;
    while (!moved) {
      moved = wc.bat();
    }
    assertNotEquals(new Coordinates(0, 0), wc.getPlayerPos());
  }
}