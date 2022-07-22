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
  String getPlayerPos();

  /**
   * Returns the gold of the player.
   *
   * @return The gold the player has.
   */
  int getPlayerGold();

  /**
   * Returns the possible moves of the player.
   *
   * @return A list of possible directions.
   */
  ArrayList<CardinalDirections> possibleMoves();

  /**
   * Moves the player in the given direction if able.
   *
   * @param direction The direction to move the player in
   */
  void movePlayer(CardinalDirections direction);

  /**
   * Returns whether the current player's position is the goal.
   *
   * @return true if the player is at the goal false if not.
   */
  boolean isGoal();

  /**
   * Simple maze print using box building characters only used for debugging does not have any
   * representative data.
   */
  void printMaze();
}
