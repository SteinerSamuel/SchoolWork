package controller;

/**
 * Maze controller for an MVC model.
 */
public interface MazeController {
  /**
   * Validates that the strings given can be used as rows and columns for the maze.
   *
   * @param colString The column string
   * @param rowString The row string
   * @return True if they are valid values for the maze, false if not
   */
  boolean validateSize(String colString, String rowString);

  /**
   * Validates strings given so they can be used as players starting position.
   *
   * @param playerxstring String representation of player starting position
   * @param playerystring String representation of player starting position
   * @return True if they are valid false if not
   */
  boolean validatePlayerPos(String playerxstring, String playerystring);

  /**
   * Validates a string representation of a seed.
   *
   * @param seedString the string representation of a seed
   * @return true if valid false if not
   */
  boolean validateSeed(String seedString);

  /**
   * Validates difficulty.
   *
   * @param batChance the string representation of the chance of a bat spawning
   * @param pitChance the string representation of the chance of a pit spawning
   * @param arrows    the string representation of the arrows the players have
   * @return true if valid false if not
   */
  boolean validateDifficulty(String batChance, String pitChance, String arrows);

  /**
   * Validates the number of walls.
   *
   * @param numberOfWallsString the string representation of the number of walls the maze should
   *                            have
   * @return true if valid else false
   */
  boolean validateNumberOfWalls(String numberOfWallsString);

  /**
   * Validates and does a move, does not returns anything, this should make calls to modify the
   * model where needed.
   *
   * @param move  String move input
   * @param shoot is this a shoot or a move
   * @param rooms number of rooms to shoot if shooting
   */
  void doMove(String move, boolean shoot, int rooms);


}

