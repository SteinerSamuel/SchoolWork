package maze;

import java.util.ArrayList;

/**
 * Maze interface, this interface provides the basic functionality fo a maze.
 */
public interface Maze {
  /**
   * Returns a string representation of the character's position on the board in x, y co-ords.
   *
   * @return A string of the players co-ordinates in the format (x, y)
   */
  Coordinates getPlayerPos();

  /**
   * Sets te player position.
   *
   * @param coord the coord to set the player position too.
   */
  void setPlayerPos(Coordinates coord);

  /**
   * Returns the most east and most south maze node coordinate of the maze.
   *
   * @return the mazes coordinates of the bottom right
   */
  Coordinates getMaxPosition();

  /**
   * Returns the gold of the player.
   *
   * @return The gold the player has.
   */
  ArrayList<CardinalDirections> possibleMoves(Coordinates coord);

  /**
   * Sets the content at given position.
   *
   * @param content  the content to add
   * @param position the position which to add the content
   */
  void setContent(Contents content, Coordinates position);

  /**
   * Gets the content at given position.
   *
   * @param position the position which to get the content from.
   * @return The content of the cell in the position given.
   */
  Contents getContent(Coordinates position);

  /**
   * Gets the cell in the direction given from the coords given.
   *
   * @param coord coordinates ot start at
   * @param cd    the direction to go
   * @return the coordinates after that.
   */
  Coordinates getAdjacent(Coordinates coord, CardinalDirections cd);

  /**
   * Retrunns the contents of all adjacent tiles which the player can move.
   *
   * @return a list of the content of the adjacent tiles
   */
  ArrayList<Contents> getAllAdjacentContent();

  /**
   * Shoots an arrow in a direction.
   *
   * @param cd direction to shoot the arrow
   * @return true if the arrow hits the wumpus false if it doesn't
   */
  boolean shootArrow(CardinalDirections cd, int rooms);

  /**
   * Returns the value of how many arrows the player has left.
   *
   * @return number of arrows
   */
  int getPlayerArrows();
}
