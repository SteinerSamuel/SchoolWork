package maze;

import java.util.ArrayList;
import java.util.Random;

/**
 * Implementation of Maze Controller.
 */
public class WumpusController implements MazeController {
  Maze maze;

  public WumpusController(int rows, int cols, boolean perfectFlag, boolean wrappedFlag,
                          int playerX, int playerY, int seed, int pitChance, int batChance) {
    this.maze = new MazeImpl(rows, cols, perfectFlag, wrappedFlag, playerX,
            playerY, seed, pitChance, batChance);
  }

  @Override
  public Coordinates movePlayer(CardinalDirections cd) {
    if (maze.possibleMoves(maze.getPlayerPos()).contains(cd)) {
      maze.setPlayerPos(maze.getAdjacent(maze.getPlayerPos(), cd));
      return maze.getPlayerPos();
    } else {
      throw new IllegalArgumentException("This is an illegal move!");
    }
  }

  @Override
  public Coordinates getPlayerPos() {
    return maze.getPlayerPos();
  }

  @Override
  public ArrayList<CardinalDirections> getPlayerMoves() {
    return maze.possibleMoves(maze.getPlayerPos());
  }

  @Override
  public ArrayList<Contents> getAdjacent() {
    return maze.getAllAdjacentContent();
  }

  @Override
  public boolean shoot(CardinalDirections cd, int rooms) {
    return maze.shootArrow(cd, rooms);
  }

  @Override
  public Contents getContent() {
    return maze.getContent(maze.getPlayerPos());
  }

  @Override
  public boolean bat() {
    int number = new Random().nextInt(2);
    if (number == 0) {
      return false;
    } else {
      maze.setPlayerPos(new Coordinates(
              new Random().nextInt(maze.getMaxPosition().getxCoordinates()),
              new Random().nextInt(maze.getMaxPosition().getyCoordinates())));
      return true;
    }
  }

  @Override
  public boolean emptyQuiver() {
    return maze.getPlayerArrows() == 0;
  }
}
