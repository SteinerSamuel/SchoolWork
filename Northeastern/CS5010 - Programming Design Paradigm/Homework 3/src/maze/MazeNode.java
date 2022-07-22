package maze;

import java.util.ArrayList;

/**
 * Maze node implementation.
 */
public interface MazeNode extends Comparable<MazeNode> {
  /**
   * Adds a wall to the Maze Node, removes the wall if the wall is already there.
   *
   * @param d The direction the wall is located in the node.
   */
  void setWall(CardinalDirections d);

  ArrayList<CardinalDirections> getWalls();

  Contains getContent();

  void setContent(Contains content);
}
