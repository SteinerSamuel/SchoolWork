package model;

import disjointset.DisjointSetImpl;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;


/**
 * Implementation of Maze this allows for both perfect and imperfect mazes as well as wrapping mazes
 * and non wrapping mazes, and any combination of those.
 */
public class HuntTheWumpus implements Maze {
  private final MazeNode[][] mazeBoard;
  private final int rows;
  private final int cols;
  private final boolean wrapping;
  private final Random random = new Random();
  private boolean wumpusDead = false;
  private Coordinates playerPos;
  private Coordinates playerPos2;
  private int playersArrows;
  private int playersArrows2;
  private boolean player2Turn;


  /**
   * Constructor for a maze. The maze built with this constructor has x coords going from left to
   * right and y coords going from top to bottom.
   *
   * @param rows     number of rows for the maze
   * @param col      number of columns for the maze
   * @param wrapping true if you want the maze to wrap false if you do not
   * @param playerx  the players starting position on the x direction
   * @param playery  the player starting position on the y direction
   * @param seed     the seed for the maze generation(This is used for random)
   * @param pitChance the chance of a pit spawning in a room
   * @param batChance the chance of a bat spawning in a room
   * @param playerArrows The number of arrows a player starts with
   * @param n the number of walls the game should have
   * @param twoPlayers sets the game to 2 player mode.
   */
  public HuntTheWumpus(int rows, int col, boolean wrapping,
                  int playerx, int playery, int seed, int pitChance, int batChance,
                       int playerArrows, int n, boolean twoPlayers) {

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
    } else if (pitChance < 0) {
      throw new IllegalArgumentException("Pit chance cannot be less than 0");
    } else if (batChance < 0) {
      throw new IllegalArgumentException("Bat chance cannot be less than 0");
    } else if (pitChance + batChance > 100) {
      throw new IllegalArgumentException(
              "Pit chance and bat chance combined must not be greater than 100.");
    }

    this.playerPos = new Coordinates(playerx, playery);
    if (twoPlayers) {
      this.playerPos2 = new Coordinates(playerx, playery);
      this.playersArrows2 = playerArrows;
    }
    this.wrapping = wrapping;
    this.rows = rows;
    this.cols = col;
    this.random.setSeed(seed);
    this.playersArrows = playerArrows;

    // Generate Maze
    this.mazeBoard = new MazeNodeImpl[col][rows];
    ArrayList<Wall> edges = new ArrayList<>();
    ArrayList<Wall> walls = new ArrayList<>();
    DisjointSetImpl<MazeNode> mazeDisjoint = new DisjointSetImpl<>();

    // Create maze board and add all edges
    for (int x = 0; x < this.cols; x++) {
      for (int y = 0; y < this.rows; y++) {
        MazeNode tmp = new MazeNodeImpl(Contents.NOTHING);
        this.mazeBoard[x][y] = tmp;
        mazeDisjoint.addSet(tmp);
        if (x > 0) {
          edges.add(new Wall(x, y, x - 1, y, CardinalDirections.WEST));
        }
        if (y > 0) {
          edges.add(new Wall(x, y, x, y - 1, CardinalDirections.NORTH));
        }

        // If wrapping add edges on the outside of the board
        if (wrapping) {
          if (x == 0) {
            edges.add(new Wall(x, y,  this.cols -  1,  y, CardinalDirections.WEST));
          }
          if (y == 0) {
            edges.add(new Wall(x, y, x, this.rows - 1, CardinalDirections.NORTH));
          }
        }
      }
    }

    // Validation of n if generating non perfect board, this value does not matter otherwise.
    int upperBound = edges.size() - (rows * col) + 1;
    if (n < 1 || n > upperBound) {
      throw new IllegalArgumentException(String.format("number of walls must be between 1 and %d",
              upperBound));
    }

    // Randomise the edges
    Collections.shuffle(edges, random);

    // Goes through all edges and adds them to a disjoint set if no changes are made to the disjoint
    // set then the wall will remain up
    for (Wall w : edges) {
      if (!mazeDisjoint.mergeSet(this.mazeBoard[w.getPosition().getXcoordinates()][w.getPosition()
              .getYcoordinates()],
              this.mazeBoard[w.getDeltaPosition().getXcoordinates()][w.getDeltaPosition()
                      .getYcoordinates()])) {
        walls.add(w);
      }
    }

    // Remove walls till the maze is formed.
    Collections.shuffle(walls, random);
    while (walls.size() > upperBound) {
      walls.remove(walls.size() - 1);
    }

    // If the board does not wrap around add walls to the edge of the board
    if (!wrapping) {
      for (int x = 0; x < this.cols; x++) {
        walls.add(new Wall(x, 0, x, this.rows - 1, CardinalDirections.NORTH));
      }
      for (int y = 0; y < rows; y++) {
        walls.add(new Wall(0, y, this.cols - 1, y, CardinalDirections.WEST));
      }
    }


    // Add the walls to the nodes this is helpful for generating the board representation in the
    // future and also is helpful for generating moves.
    for (Wall w : walls) {
      this.mazeBoard[w.getPosition().getXcoordinates()][w.getPosition().getYcoordinates()]
              .setWall(w.getDir());
      this.mazeBoard[w.getDeltaPosition().getXcoordinates()][w.getDeltaPosition().getYcoordinates()]
              .setWall(w.getDir().opposite());
    }


    // Add the wumpus to the maze.
    boolean wumpusNotAdded = true; 
    while (wumpusNotAdded) {
      int randX = random.nextInt(this.cols);
      int randY = random.nextInt(this.rows);
      
      // If the random coordinates are a room and are not the starting position of the player. 
      if (this.mazeBoard[randX][randY].isRoom()
              && !(new Coordinates(randX, randY).equals(playerPos))) {
        this.mazeBoard[randX][randY].setContent(Contents.WUMPUS);
        wumpusNotAdded = false;
      }
    }

    // Rolling contents of rooms.
    for (int y = 0; y < this.rows; y++) {
      for (int x = 0; x < this.cols; x++) {
        // check if coord is a room and has nothing, and is not the player starting pos.
        if (this.mazeBoard[x][y].isRoom()
                && this.mazeBoard[x][y].getContent().equals(Contents.NOTHING)
                && !(new Coordinates(x, y).equals(playerPos))) {
          Contents contents = rollContent(Contents.BATS, batChance, 0);
          if (contents.equals(Contents.NOTHING)) {
            contents = rollContent(Contents.PIT, pitChance, batChance);
          }
          this.mazeBoard[x][y].setContent(contents);
        }
      }
    }
  }

  @Override
  public Coordinates getPlayerPos(boolean player2) {
    if (player2) {
      return playerPos2;
    }
    return playerPos;
  }

  @Override
  public void setPlayerPos(Coordinates coord, boolean player2) {
    if (player2) {
      playerPos2 = coord;
    } else {
      playerPos = coord;
    }
  }

  @Override
  public Coordinates getMaxPosition() {
    return new Coordinates(cols - 1, rows - 1);
  }

  @Override
  public ArrayList<CardinalDirections> possibleMoves(Coordinates coord) {
    ArrayList<CardinalDirections> walls = this.mazeBoard[coord.getXcoordinates()]
            [coord.getYcoordinates()].getWalls();

    ArrayList<CardinalDirections> directions = new ArrayList<>(
            Arrays.asList(CardinalDirections.NORTH,
                    CardinalDirections.SOUTH, CardinalDirections.EAST, CardinalDirections.WEST));

    for (CardinalDirections cd : walls) {
      directions.remove(cd);
    }

    return directions;
  }

  @Override
  public Contents getContent(Coordinates position) {
    return this.mazeBoard[position.getXcoordinates()][position.getYcoordinates()].getContent();
  }

  @Override
  public Coordinates getAdjacent(Coordinates pos, CardinalDirections direction) {
    if (wrapping) {
      if (direction.equals(CardinalDirections.WEST) && pos.getXcoordinates() == 0) {
        return new Coordinates(cols - 1, pos.getYcoordinates());
      } else if (direction.equals(CardinalDirections.EAST)
              && pos.getXcoordinates() == this.cols - 1) {
        return new Coordinates(0, pos.getYcoordinates());
      } else if (direction.equals(CardinalDirections.NORTH)
              && pos.getYcoordinates() == 0) {
        return new Coordinates(pos.getXcoordinates(), rows - 1);
      } else if (direction.equals(CardinalDirections.SOUTH)
              && pos.getYcoordinates() == this.rows - 1) {
        return new Coordinates(pos.getXcoordinates(), 0);
      }
    }
    switch (direction) {
      case WEST:
        return new Coordinates(pos.getXcoordinates() - 1,
                pos.getYcoordinates());
      case EAST:
        return new Coordinates(pos.getXcoordinates() + 1,
                pos.getYcoordinates());
      case NORTH:
        return new Coordinates(pos.getXcoordinates(),
                pos.getYcoordinates() - 1);
      case SOUTH:
        return new Coordinates(pos.getXcoordinates(),
                pos.getYcoordinates() + 1);
      default:
        return pos;

    }
  }

  @Override
  public ArrayList<Contents> getAllAdjacentContent(boolean player2) {
    ArrayList<CardinalDirections> cds;
    if (player2) {
      cds = possibleMoves(this.playerPos2);
    } else {
      cds = possibleMoves(this.playerPos);
    }
    ArrayList<Contents> adjContent = new ArrayList<>();
    for (CardinalDirections cd : cds) {
      Coordinates coords = getAdjacent(getPlayerPos(player2), cd);
      adjContent.add(mazeBoard[coords.getXcoordinates()][coords.getYcoordinates()].getContent());
    }
    adjContent.removeAll(Arrays.asList(Contents.NOTHING, Contents.BATS));
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
  public int getPlayerArrows(boolean player2) {
    if (player2) {
      return playersArrows2;
    } else {
      return playersArrows;
    }
  }


  @Override
  public Coordinates movePlayer(CardinalDirections cd, boolean player2) {
    if (possibleMoves(getPlayerPos(player2)).contains(cd)) {
      setPlayerPos(getAdjacent(getPlayerPos(player2), cd), player2);
      if (mazeBoard[getPlayerPos(player2).getXcoordinates()]
              [getPlayerPos(player2).getYcoordinates()].getContent().equals(Contents.BATS)) {
        if (random.nextInt() % 2 == 1) {
          setPlayerPos(new Coordinates(random.nextInt(this.cols), random.nextInt(this.rows)),
                  player2);
        }
      }
      return getPlayerPos(player2);
    } else {
      throw new IllegalArgumentException("This is an illegal move!");
    }
  }


  @Override
  public void shootArrow(CardinalDirections cd, int rooms, boolean player2) {
    // Checks if the player has arrows
    if (playersArrows != 0) {
      // arrow starts where the playe ris
      Coordinates startingPos = getPlayerPos(player2);
      while (possibleMoves(startingPos).contains(cd) && rooms > 0) {
        startingPos = getAdjacent(startingPos, cd);
        if (this.mazeBoard[startingPos.getXcoordinates()][startingPos.getYcoordinates()].isRoom()) {
          rooms--;
        }
      }
      if (this.mazeBoard[startingPos.getXcoordinates()][startingPos.getYcoordinates()].getContent()
              .equals(Contents.WUMPUS)) {
        this.wumpusDead = true;
        this.player2Turn = player2;
      }
    }
    playersArrows--;
  }

  @Override
  public GameState checkGameState(boolean player2) {
    boolean player2State;
    boolean player1State = (mazeBoard[this.playerPos.getXcoordinates()]
            [this.playerPos.getYcoordinates()].getContent().equals(Contents.WUMPUS)
            || mazeBoard[this.playerPos.getXcoordinates()]
            [this.playerPos.getYcoordinates()].getContent().equals(Contents.PIT)
            || this.playersArrows == 0);
    if (player2) {
      System.out.println(mazeBoard[this.playerPos2.getXcoordinates()]
              [this.playerPos2.getYcoordinates()].getContent().equals(Contents.WUMPUS));
      System.out.println(mazeBoard[this.playerPos2.getXcoordinates()]
              [this.playerPos2.getYcoordinates()].getContent().equals(Contents.PIT));
      System.out.println(this.playersArrows2 == 0);
      player2State = (mazeBoard[this.playerPos2.getXcoordinates()]
              [this.playerPos2.getYcoordinates()].getContent().equals(Contents.WUMPUS)
              || mazeBoard[this.playerPos2.getXcoordinates()]
              [this.playerPos2.getYcoordinates()].getContent().equals(Contents.PIT)
              || this.playersArrows2 == 0);
    } else {
      player2State = true;
    }

    if (wumpusDead) {
      if (player2Turn) {
        return GameState.PLAYER2WIN;
      } else {
        return GameState.PLAYER1WIN;
      }
    } else if (player1State && player2State) {
      return GameState.FULLOSS;
    } else if (player1State) {
      return GameState.PLAYER1LOSS;
    }  else if (player2State) {
      return GameState.PLAYER2LOSS;
    } else {
      return GameState.INPROGRESS;
    }
  }
}