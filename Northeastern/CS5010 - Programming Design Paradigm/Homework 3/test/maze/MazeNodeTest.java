package maze;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.assertEquals;

/**
 * Test the interface for maze node.
 */
public class MazeNodeTest {
  MazeNode testNode;

  @Before
  public void setUp() {
    testNode = new MazeNodeImpl(Contains.NOTHING);
  }

  @Test
  public void testSetGetWall() {
    ArrayList<CardinalDirections> testWall = new ArrayList<CardinalDirections>();

    assertEquals(testWall, testNode.getWalls());
    testNode.setWall(CardinalDirections.NORTH);

    testWall.add(CardinalDirections.NORTH);

    assertEquals(testWall, testNode.getWalls());
  }

  @Test
  public void testSetGetContent() {
    assertEquals(Contains.NOTHING, testNode.getContent());

    testNode.setContent(Contains.GOLD);

    assertEquals(Contains.GOLD, testNode.getContent());
  }
}