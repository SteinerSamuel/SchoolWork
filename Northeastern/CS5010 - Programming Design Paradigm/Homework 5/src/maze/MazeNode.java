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

  /**
   * Gets the walls of the nodes.
   *
   * @return An array of the walls of the node
   */
  ArrayList<CardinalDirections> getWalls();

  /**
   * Gets the contents of the room.
   *
   * @return the contents of the room
   */
  Contents getContent();

  /**
   * Sets the contents of the room.
   *
   * @param content what is in the room.
   */
  void setContent(Contents content);
}
