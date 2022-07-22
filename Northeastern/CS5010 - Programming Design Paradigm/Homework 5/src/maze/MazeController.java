package maze;

import java.util.ArrayList;

/**
 * Maze controller for an MVC model.
 */
public interface MazeController {
  /**
   * Moves the player in the maze in the cardinal direction given.
   *
   * @param cd the cardinal direction to move the player in
   * @return the new coordinates of the player
   */
  Coordinates movePlayer(CardinalDirections cd);

  /**
   * Gets the coordinates of the players pos.
   *
   * @return The players position
   */
  Coordinates getPlayerPos();

  /**
   * Gets the moves available to the player.
   *
   * @return the moves which the player can take.
   */
  ArrayList<CardinalDirections> getPlayerMoves();

  /**
   * Returns the list of the contents of the rooms around the player.
   *
   * @return a list of content in the rooms around the player.
   */
  ArrayList<Contents> getAdjacent();

  /**
   * Shoots an arrow.
   *
   * @param cd    the cardinal direction to shoot the arrow down
   * @param rooms how many rooms down to shoot the arrow
   * @return true if it hits the wumpus false otherwise
   */
  boolean shoot(CardinalDirections cd, int rooms);

  /**
   * Gets the content of the current square.
   *
   * @return content of the current player pos
   */
  Contents getContent();

  /**
   * Bat method.
   *
   * @return true if the bats move the player, false if the players dodge the bats.
   */
  boolean bat();

  /**
   * Returns true if player runs out of arrows.
   *
   * @return true if quiver is empty false if not
   */
  boolean emptyQuiver();
}
