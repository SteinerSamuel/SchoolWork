package model;

import java.util.ArrayList;

/**
 * Maze interface, this interface provides the basic functionality fo a maze.
 */
public interface Maze {
  /**
   * Returns a string representation of the character's position on the board in x, y co-ords.
   *
   * @param player2 whether its players 2 turn or not
   * @return A string of the players co-ordinates in the format (x, y)
   */
  Coordinates getPlayerPos(boolean player2);

  /**
   * Sets te player position.
   *
   * @param player2 whether its players 2 turn or not
   * @param coord the coord to set the player position too.
   */
  void setPlayerPos(Coordinates coord, boolean player2);

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
   * @param player2 whether its players 2 turn or not
   * @return a list of the content of the adjacent tiles
   */
  ArrayList<Contents> getAllAdjacentContent(boolean player2);

  /**
   * Shoots an arrow in a direction.
   *
   * @param cd direction to shoot the arrow
   * @param rooms  number of rooms to shoot
   * @param player2  whether its players 2 turn or not
   *
   */
  void shootArrow(CardinalDirections cd, int rooms, boolean player2);

  /**
   * Returns the value of how many arrows the player has left.
   *
   * @param player2 whether its players 2 turn or not.
   * @return number of arrows
   */
  int getPlayerArrows(boolean player2);

  /**
   * Moves the player in a given direction.
   *
   * @param cd the direction to move the player.
   * @param player2 whether its players2 turn or not
   * @return the Coordinates of the players new position.
   */
  Coordinates movePlayer(CardinalDirections cd, boolean player2);

  /**
   * Checks the game state of the current board.
   *
   * @param player2 whether its players 2 turn or not
   * @return the game state.
   */
  GameState checkGameState(boolean player2);
}
