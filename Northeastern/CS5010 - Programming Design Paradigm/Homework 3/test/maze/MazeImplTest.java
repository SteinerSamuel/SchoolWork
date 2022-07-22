package maze;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Test the Maze interface.
 */
public class MazeImplTest {
  private int seed;

  @Before
  public void setUp() {
    seed = 123; //This removes the randomness from the maze generation so we can test functionality
  }

  @Test
  public void testMazeGeneration() {
    // Perfect mazes ignore the value of 0
    Maze testMaze = new MazeImpl(3, 3, 0, true, false, 10, 0, 0, 2, 2, seed);
    testMaze = new MazeImpl(3, 3, -123, true, false, 10, 0, 0, 2, 2, seed);

    // should throw errors when invalid values are given for rows, col, goldValue, player co-ords
    // and goal co-ords

    //Column illegal value test
    try {
      testMaze = new MazeImpl(3, 0, 0, true, false, 10, 0, 0, 0, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }
    try {
      testMaze = new MazeImpl(3, -10, 0, true, false, 10, 0, 0, 0, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    //row illegal value test
    try {
      testMaze = new MazeImpl(-100, 10, 0, true, false, 10, 0, 0, 0, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    try {
      testMaze = new MazeImpl(0, 10, 0, true, false, 10, 0, 0, 0, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    //gold illegal value test
    try {
      testMaze = new MazeImpl(10, 10, 0, true, false, 0, 0, 0, 0, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    try {
      testMaze = new MazeImpl(10, 10, 0, true, false, -10, 0, 0, 0, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    //player pos illegal value test
    try {
      testMaze = new MazeImpl(10, 10, 0, true, false, 10, 10, 0, 0, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    try {
      testMaze = new MazeImpl(10, 10, 0, true, false, 10, -1, 0, 0, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    try {
      testMaze = new MazeImpl(10, 10, 0, true, false, 10, 0, 12, 0, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    try {
      testMaze = new MazeImpl(10, 10, 0, true, false, 10, 0, -10, 0, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    try {
      testMaze = new MazeImpl(10, 10, 0, true, false, 10, 0, 0, 123, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    try {
      testMaze = new MazeImpl(10, 10, 0, true, false, 10, -0, 0, -123, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    try {
      testMaze = new MazeImpl(10, 10, 0, true, false, 10, 0, 0, 0, 123, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    try {
      testMaze = new MazeImpl(10, 10, 0, true, false, 10, 0, 0, 0, -123, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }


  }

  @Test
  public void testMazeGenerationNonPerfect() {
    Maze testMaze;

    // im-perfect illegal value test 0 remaining walls
    try {
      testMaze = new MazeImpl(10, 10, 0, false, false, 10, 0, 0, 0, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }
    // im-perfect illegal value test negative remaining walls
    try {
      testMaze = new MazeImpl(10, 10, -10, false, false, 10, 0, 0, 0, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }
    // im-perfect illegal value test too many remaining walls
    try {
      testMaze = new MazeImpl(10, 10, 200, false, false, 10, 0, 0, 0, 2, seed);
      fail("IllegalArgumentException expected.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }
  }

  @Test
  public void testGetPlayerPos() {
    Maze testMaze = new MazeImpl(3, 3, 0, true, false, 10, 0, 0, 2, 2, seed);

    assertEquals("(0, 0)", testMaze.getPlayerPos());
  }

  @Test
  public void testPossibleMoves() {
    Maze testMaze = new MazeImpl(3, 3, 0, true, false, 10, 0, 0, 2, 2, seed);
    ArrayList<CardinalDirections> testList = new ArrayList<CardinalDirections>();

    testList.add(CardinalDirections.SOUTH);
    assertEquals(testList,
            testMaze.possibleMoves());
  }


  @Test
  public void testMoveCharacterNonWrapping() {
    Maze testMaze = new MazeImpl(3, 3, 0, true, false, 10, 0, 0, 2, 2, seed);

    assertEquals("(0, 0)", testMaze.getPlayerPos());

    testMaze.movePlayer(CardinalDirections.SOUTH);
    assertEquals("(0, 1)", testMaze.getPlayerPos());
  }

  @Test
  public void testMoveCharacterWrapping() {
    Maze testMaze = new MazeImpl(3, 3, 0, true, true, 10, 0, 0, 2, 2, seed);

    assertEquals("(0, 0)", testMaze.getPlayerPos());

    testMaze.movePlayer(CardinalDirections.WEST);
    assertEquals("(2, 0)", testMaze.getPlayerPos());
  }

  @Test
  public void testGetIsGoal() {
    Maze testMaze = new MazeImpl(3, 3, 0, true, false, 10, 0, 0, 2, 2, seed);

    assertFalse(testMaze.isGoal());

    testMaze.movePlayer(CardinalDirections.SOUTH);
    testMaze.movePlayer(CardinalDirections.EAST);
    testMaze.movePlayer(CardinalDirections.SOUTH);
    testMaze.movePlayer(CardinalDirections.EAST);

    assertTrue(testMaze.isGoal());
  }

  @Test
  public void testGetPlayerGold() {
    Maze testMaze = new MazeImpl(3, 3, 0, true, false, 10, 0, 0, 2, 2, 555);

    assertEquals(0, testMaze.getPlayerGold());

    testMaze.movePlayer(CardinalDirections.EAST);
    testMaze.movePlayer(CardinalDirections.EAST);
    assertEquals(10, testMaze.getPlayerGold());
  }

}