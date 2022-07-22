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
  private final boolean wrapping;
  private final Random random = new Random();
  private Coordinates playerPos = new Coordinates(0, 0);
  private int playersArrows;

  /**
   * Constructor for a maze. The maze built with this constructor has x coords going from left to
   * right and y coords going from top to bottom.
   *
   * @param rows     number of rows for the maze
   * @param col      number of columns for the maze
   * @param perfect  true if you want to generate a perfect maze false if you want an imperfect
   *                 maze
   * @param wrapping true if you want the maze to wrap false if you do not
   * @param playerx  the players starting position on the x direction
   * @param playery  the player starting position on the y direction
   * @param seed     the seed for the maze generation(This is used for random)
   */
  public MazeImpl(int rows, int col, boolean perfect, boolean wrapping,
                  int playerx, int playery, int seed, int pitChance, int batChance) {

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
    }

    this.playerPos.setyCoordinates(playery);
    this.playerPos.setxCoordinates(playerx);
    this.wrapping = wrapping;
    this.rows = rows;
    this.cols = col;
    this.random.setSeed(seed);

    int wumpusX = random.nextInt(cols);
    int wumpusY = random.nextInt(rows);

    // Generate Maze
    this.mazeBoard = new MazeNodeImpl[rows][col];
    ArrayList<Wall> edges = new ArrayList<>();
    ArrayList<Wall> walls = new ArrayList<>();
    DisjointSetImpl<MazeNode> mazeDisjoint = new DisjointSetImpl<>();

    // Create maze board and add all edges
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < col; j++) {
        Contents content = rollContent(Contents.PIT, pitChance, 0);
        if (content == Contents.NOTHING) {
          content = rollContent(Contents.BATS, batChance, pitChance);
        }
        if (i == playerx && j == playery) {
          content = Contents.NOTHING;
        }
        if (i == wumpusX && j == wumpusY) {
          content = Contents.WUMPUS;
        }
        MazeNode tmp = new MazeNodeImpl(content);
        this.mazeBoard[j][i] = tmp;
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
      while (walls.size() > upperBound) {
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
      this.mazeBoard[w.xpos][w.ypos].setWall(w.dir);
      this.mazeBoard[w.dx][w.dy].setWall(w.dir.opposite());
    }
  }

  @Override
  public Coordinates getPlayerPos() {
    return playerPos;
  }

  @Override
  public void setPlayerPos(Coordinates coord) {
    playerPos = coord;
  }

  @Override
  public Coordinates getMaxPosition() {
    return new Coordinates(cols - 1, rows - 1);
  }

  @Override
  public ArrayList<CardinalDirections> possibleMoves(Coordinates coord) {
    ArrayList<CardinalDirections> walls = this.mazeBoard[coord.getxCoordinates()]
            [coord.getyCoordinates()].getWalls();

    ArrayList<CardinalDirections> directions = new ArrayList<>(
            Arrays.asList(CardinalDirections.NORTH,
                    CardinalDirections.SOUTH, CardinalDirections.EAST, CardinalDirections.WEST));

    for (CardinalDirections cd : walls) {
      directions.remove(cd);
    }

    return directions;
  }

  @Override
  public void setContent(Contents content, Coordinates position) {
    this.mazeBoard[position.getxCoordinates()][position.getyCoordinates()].setContent(content);
  }

  @Override
  public Contents getContent(Coordinates position) {
    return this.mazeBoard[position.getxCoordinates()][position.getyCoordinates()].getContent();
  }

  @Override
  public Coordinates getAdjacent(Coordinates pos, CardinalDirections direction) {
    if (wrapping) {
      if (direction.equals(CardinalDirections.WEST) && pos.getxCoordinates() == 0) {
        return new Coordinates(cols - 1, pos.getyCoordinates());
      } else if (direction.equals(CardinalDirections.EAST)
              && pos.getxCoordinates() == this.cols - 1) {
        return new Coordinates(0, pos.getyCoordinates());
      } else if (direction.equals(CardinalDirections.NORTH)
              && pos.getyCoordinates() == 0) {
        return new Coordinates(pos.getxCoordinates(), rows - 1);
      } else if (direction.equals(CardinalDirections.SOUTH)
              && pos.getyCoordinates() == this.rows - 1) {
        return new Coordinates(pos.getxCoordinates(), 0);
      }
    }
    switch (direction) {
      case WEST:
        return new Coordinates(pos.getxCoordinates() - 1,
                pos.getyCoordinates());
      case EAST:
        return new Coordinates(pos.getxCoordinates() + 1,
                pos.getyCoordinates());
      case NORTH:
        return new Coordinates(pos.getxCoordinates(),
                pos.getyCoordinates() - 1);
      case SOUTH:
        return new Coordinates(pos.getxCoordinates(),
                pos.getyCoordinates() + 1);
      default:
        return pos;

    }
  }

  @Override
  public ArrayList<Contents> getAllAdjacentContent() {
    ArrayList<CardinalDirections> cds = possibleMoves(this.playerPos);
    ArrayList<Contents> adjContent = new ArrayList<>();
    for (CardinalDirections cd : cds) {
      Coordinates coords = getAdjacent(getPlayerPos(), cd);
      adjContent.add(mazeBoard[coords.getxCoordinates()][coords.getyCoordinates()].getContent());
    }
    return adjContent;
  }

  /**
   * Helper class to determine the content of a room.
   *
   * @return a Contains of what the room will contain
   */
  private Contents rollContent(Contents content, int chance, int offset) {
    int r = this.random.nextInt(100 - offset);
    if (r < chance) {
      return content;
    } else {
      return Contents.NOTHING;
    }
  }

  @Override
  public int getPlayerArrows() {
    return playersArrows;
  }

  @Override
  public boolean shootArrow(CardinalDirections cd, int rooms) {
    boolean hit = false;
    if (playersArrows != 0) {
      Coordinates startingPos = getPlayerPos();
      while (possibleMoves(startingPos).contains(cd) && rooms > 0) {
        startingPos = getAdjacent(startingPos, cd);
        if (!getContent(startingPos).equals(Contents.NOTHING)) {
          rooms--;
        }
      }
      if (mazeBoard[startingPos.getxCoordinates()][startingPos.getyCoordinates()].getContent()
              == Contents.WUMPUS) {
        hit = true;
      }
    }
    playersArrows--;
    return hit;
  }
}