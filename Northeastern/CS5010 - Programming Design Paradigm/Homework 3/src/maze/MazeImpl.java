package maze;

import disjointset.DisjointSetImpl;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;


/**
 * Implementation of Maze this allows for both perfect and imperfect mazes as well as wrapping mazes
 * and non wrapping mazes, and any combination of those.
 */
public class MazeImpl implements Maze {
  private final MazeNode[][] mazeBoard;
  private final int rows;
  private final int cols;
  private final int[] playerPos = new int[2];
  private final int[] goalPos = new int[2];
  private final int goldValue;
  private final boolean wrapping;
  private int playersGold;
  private final Random random = new Random();

  /**
   * Constructor for a maze. The maze built with this constructor has x coords going from left to
   * right and y coords going from top to bottom.
   *
   * @param rows      number of rows for the maze
   * @param col       number of columns for the maze
   * @param n         number of remaining walls only matters if the maze is imperfect
   * @param perfect   true if you want to generate a perfect maze false if you want an imperfect
   *                  maze
   * @param wrapping  true if you want the maze to wrap false if you do not
   * @param goldValue the value of each stack of gold
   * @param playerx   the players starting position on the x direction
   * @param playery   the player starting position on the y direction
   * @param goalX     the goal position on the x direction
   * @param goalY     the goal position on the y direction
   * @param seed      the seed for the maze generation(This is used for random)
   */
  public MazeImpl(int rows, int col, int n, boolean perfect, boolean wrapping, int goldValue,
                  int playerx, int playery, int goalX, int goalY, int seed) {

    // Validation
    if (rows < 1) {
      throw new IllegalArgumentException("Rows must be greater than 0");
    } else if (col < 1) {
      throw new IllegalArgumentException("Columns must be greater than 0");
    } else if (playerx < 0 || playerx >= col) {
      throw new IllegalArgumentException("Player position x  must be between 0 inclusive and "
              + "number of columns exclusive");
    } else if (playery < 0 || playery >= rows) {
      throw new IllegalArgumentException("Player position y must be between 0 inclusive and number"
              + " of rows exclusive");
    } else if (goalX < 0 || goalX >= col) {
      throw new IllegalArgumentException("goal position x  must be between 0 inclusive and "
              + "number of columns exclusive");
    } else if (goalY < 0 || goalY >= rows) {
      throw new IllegalArgumentException("goal position y must be between 0 inclusive and number"
              + " of rows exclusive");
    } else if (goldValue < 1) {
      throw new IllegalArgumentException("Gold value must be greater than 0");
    }

    this.goldValue = goldValue;
    this.playerPos[0] = playery;
    this.playerPos[1] = playerx;
    this.playersGold = 0;
    this.goalPos[0] = goalY;
    this.goalPos[1] = goalX;
    this.wrapping = wrapping;
    this.rows = rows;
    this.cols = col;
    this.random.setSeed(seed);

    // Generate Maze
    this.mazeBoard = new MazeNodeImpl[rows][col];
    ArrayList<Wall> edges = new ArrayList<>();
    ArrayList<Wall> walls = new ArrayList<>();
    DisjointSetImpl<MazeNode> mazeDisjoint = new DisjointSetImpl<>();

    // Create maze board and add all edges
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < col; j++) {
        MazeNode tmp = new MazeNodeImpl(rollContent());
        this.mazeBoard[i][j] = tmp;
        mazeDisjoint.addSet(tmp);
        if (i > 0) {
          edges.add(new Wall(i, j, i - 1, j, CardinalDirections.NORTH));
        }
        if (j > 0) {
          edges.add(new Wall(i, j, i, j - 1, CardinalDirections.WEST));
        }

        // If wrapping add edges on the outside of the board
        if (wrapping) {
          if (i == 0) {
            edges.add(new Wall(i, j, rows - 1, j, CardinalDirections.NORTH));
          }
          if (j == 0) {
            edges.add(new Wall(i, j, i, col - 1, CardinalDirections.WEST));
          }
        }
      }
    }

    // Validation of n if generating non perfect board, this value does not matter otherwise.
    int upperBound = edges.size() - (rows * col) + 1;
    if (!perfect && (n <= 0 || n >= upperBound)) {
      throw new IllegalArgumentException("The number of remaining wall must be between 0 and "
              + "numberOfEdges - n + 1. which for a maze of " + rows + " rows and " + col
              + " columns is " + upperBound);
    }

    // Randomise the edges
    Collections.shuffle(edges, random);

    // Goes through all edges and adds them to a disjoint set if no changes are made to the disjoint
    // set then the wall will remain up
    for (Wall w : edges) {
      if (!mazeDisjoint.mergeSet(this.mazeBoard[w.ypos][w.xpos], this.mazeBoard[w.dy][w.dx])) {
        walls.add(w);
      }
    }

    // If the board is not perfect randomly remove walls from the remaining walls
    if (!perfect) {
      Collections.shuffle(walls, random);
      while (walls.size() > n) {
        walls.remove(walls.size() - 1);
      }
    }

    // If the board does not wrap around add walls to the edge of the board
    if (!wrapping) {
      for (int i = 0; i < col; i++) {
        walls.add(new Wall(0, i, rows - 1, i, CardinalDirections.NORTH));
      }
      for (int i = 0; i < rows; i++) {
        walls.add(new Wall(i, 0, i, col - 1, CardinalDirections.WEST));
      }
    }


    // Add the walls to the nodes this is helpful for generating the board representation in the
    // future and also is helpful for generating moves.
    for (Wall w : walls) {
      this.mazeBoard[w.ypos][w.xpos].setWall(w.dir);
      this.mazeBoard[w.dy][w.dx].setWall(w.dir.opposite());
    }
  }

  @Override
  public String getPlayerPos() {
    return String.format("(%d, %d)", playerPos[1], playerPos[0]);
  }

  @Override
  public int getPlayerGold() {
    return playersGold;
  }

  @Override
  public ArrayList<CardinalDirections> possibleMoves() {
    ArrayList<CardinalDirections> walls = this.mazeBoard[playerPos[0]][playerPos[1]].getWalls();
    if (walls.isEmpty()) {
      return new ArrayList<CardinalDirections>(Arrays.asList(CardinalDirections.NORTH,
              CardinalDirections.SOUTH, CardinalDirections.EAST, CardinalDirections.WEST));
    } else if (walls.size() == 1) {
      switch (walls.get(0)) {
        case NORTH:
          return new ArrayList<CardinalDirections>(Arrays.asList(CardinalDirections.SOUTH,
                CardinalDirections.EAST, CardinalDirections.WEST));
        case SOUTH:
          return new ArrayList<CardinalDirections>(Arrays.asList(CardinalDirections.NORTH,
                CardinalDirections.EAST, CardinalDirections.WEST));
        case WEST:
          return new ArrayList<CardinalDirections>(Arrays.asList(CardinalDirections.NORTH,
                CardinalDirections.SOUTH, CardinalDirections.EAST));
        case EAST:
          return new ArrayList<CardinalDirections>(Arrays.asList(CardinalDirections.NORTH,
                CardinalDirections.SOUTH, CardinalDirections.WEST));
        default:
          break;
      }
    } else if (walls.size() == 2) {
      if (walls.contains(CardinalDirections.EAST) && walls.contains(CardinalDirections.WEST)) {
        return new ArrayList<CardinalDirections>(Arrays.asList(CardinalDirections.NORTH,
                CardinalDirections.SOUTH));
      } else if (walls.contains(CardinalDirections.EAST) && walls.contains(CardinalDirections.NORTH
      )) {
        return new ArrayList<CardinalDirections>(Arrays.asList(CardinalDirections.SOUTH,
                CardinalDirections.WEST));
      } else if (walls.contains(CardinalDirections.EAST) && walls.contains(CardinalDirections.SOUTH
      )) {
        return new ArrayList<CardinalDirections>(Arrays.asList(CardinalDirections.NORTH,
                CardinalDirections.WEST));
      } else if (walls.contains(CardinalDirections.NORTH) && walls.contains(CardinalDirections.WEST
      )) {
        return new ArrayList<CardinalDirections>(Arrays.asList(CardinalDirections.SOUTH,
                CardinalDirections.EAST));
      } else if (walls.contains(CardinalDirections.SOUTH) && walls.contains(CardinalDirections.WEST
      )) {
        return new ArrayList<CardinalDirections>(Arrays.asList(CardinalDirections.NORTH,
                CardinalDirections.EAST));
      } else {
        return new ArrayList<CardinalDirections>(Arrays.asList(CardinalDirections.EAST,
                CardinalDirections.WEST));
      }
    } else if (walls.size() == 3) {
      if (!walls.contains(CardinalDirections.EAST)) {
        return new ArrayList<CardinalDirections>(Collections.singletonList(CardinalDirections.EAST)
        );
      } else if (!walls.contains(CardinalDirections.WEST)) {
        return new ArrayList<CardinalDirections>(Collections.singletonList(CardinalDirections.WEST)
        );
      } else if (!walls.contains(CardinalDirections.NORTH)) {
        return new ArrayList<CardinalDirections>(Collections.singletonList(CardinalDirections.NORTH)
        );
      } else {
        return new ArrayList<CardinalDirections>(Collections.singletonList(CardinalDirections.SOUTH)
        );
      }
    }
    return null;
  }

  @Override
  public void movePlayer(CardinalDirections direction) {
    // Check if the move is possible
    if (possibleMoves().contains(direction)) {

      // Move the player
      if (this.wrapping) {
        if (direction.equals(CardinalDirections.WEST) && this.playerPos[1] == 0) {
          this.playerPos[1] = cols - 1;
        } else if (direction.equals(CardinalDirections.EAST)
                && this.playerPos[1] == this.cols - 1) {
          this.playerPos[1] = 0;
        } else if (direction.equals(CardinalDirections.NORTH) && this.playerPos[0] == 0) {
          this.playerPos[0] = rows - 1;
        } else if (direction.equals(CardinalDirections.SOUTH)
                && this.playerPos[0] == this.rows - 1) {
          this.playerPos[0] = 0;
        } else {
          moveHelper(direction);
        }
      } else {
        moveHelper(direction);
      }

      // Check the room for content
      Contains content = mazeBoard[playerPos[0]][playerPos[1]].getContent();

      switch (content) {
        case GOLD:
          System.out.println("The location has gold of value: " + goldValue);
          playersGold += goldValue;
          mazeBoard[playerPos[0]][playerPos[1]].setContent(Contains.NOTHING);
          break;
        case THIEF:
          System.out.println("The location has a thief he steals 10% of your gold");
          playersGold = (9 * playersGold / 10);
          break;
        default:
          // Do nothing if there is nothing in the room
          break;
      }

    } else {
      throw new IllegalArgumentException("Not a possible move");
    }
  }

  /**
   * Helper function used to remove duplicate code block.
   *
   * @param direction The direction to move
   */
  private void moveHelper(CardinalDirections direction) {
    switch (direction) {
      case WEST:
        this.playerPos[1]--;
        break;
      case EAST:
        this.playerPos[1]++;
        break;
      case NORTH:
        this.playerPos[0]--;
        break;
      case SOUTH:
        this.playerPos[0]++;
        break;
      default:
        break;
    }
  }

  @Override
  public boolean isGoal() {
    return this.goalPos[0] == this.playerPos[0] && this.goalPos[1] == this.playerPos[1];
  }

  /**
   * Helper class to determine the content of a room.
   *
   * @return a Contains of what the room will contain
   */
  private Contains rollContent() {
    int r = this.random.nextInt(100);
    if (r < 10) {
      return Contains.THIEF;
    } else if (r < 30) {
      return Contains.GOLD;
    } else {
      return Contains.NOTHING;
    }
  }

  @Override
  public void printMaze() {
    for (MazeNode[] m : mazeBoard) {
      for (MazeNode n : m) {
        System.out.print(n);
      }
      System.out.println();
    }
  }
}