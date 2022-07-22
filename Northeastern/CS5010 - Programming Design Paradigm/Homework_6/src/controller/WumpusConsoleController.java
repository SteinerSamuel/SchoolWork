package controller;

import java.util.Random;
import model.CardinalDirections;
import model.Contents;
import model.GameState;
import model.HuntTheWumpus;
import model.Maze;
import view.WumpusConsoleView;

/**
 * Implementation of the Maze controller meant for use with the conosle view.
 */
public class WumpusConsoleController implements MazeController {

  private WumpusConsoleView view;
  private Maze model;
  private int validatedRow;
  private int validatedCol;
  private int validatedx;
  private int validatedy;
  private int validatedNumbefOfWalls;
  private int validatedBat;
  private int validatedPit;
  private int validatedArrow;
  private int validatedSeed = new Random().nextInt();
  private int validatedRoom = 0;
  private boolean validatedShoot = false;
  private boolean validatedWrapper;
  private boolean validatedPlayer;
  private boolean turnPlayer = false;

  /**
   * Console controller constructor takes the view.
   *
   * @param view the view to be used with this controller.
   */
  public WumpusConsoleController(WumpusConsoleView view) {
    this.view = view;

    this.view.pushMessage("Welcome to Wumpus World.");
    this.view.pushMessage("Let's get your game setup");
    String colString = this.view.getUserInputSetting("Number of Columns");
    String rowString = this.view.getUserInputSetting("Number of Rows");
    while (!validateSize(colString, rowString)) {
      colString = this.view.getUserInputSetting("Number of Columns,"
              + " input must be a integer above 0");
      rowString = this.view.getUserInputSetting("Number of Rows, input must be an integer above 0");
    }

    String userx = this.view.getUserInputSetting("On what column should the player start");
    String usery = this.view.getUserInputSetting("On what row should the player start");
    while (!validatePlayerPos(userx, usery)) {
      userx = this.view.getUserInputSetting("On what column should the player start");
      usery = this.view.getUserInputSetting("On what row should the player start");
    }

    String numberOfWallString = this.view.getUserInputSetting(
            String.format("How many walls should the maze has choose a number from 1 to %d",
                    (((validatedRow * validatedCol) * 2) - (validatedRow * validatedCol - 1
                            + (validatedCol + validatedRow))))
    );
    while (!validateNumberOfWalls(numberOfWallString)) {
      numberOfWallString = this.view.getUserInputSetting(
              String.format("PLease make sure the input is a valid integer. \n"
                              + "How many walls should the maze has choose a number from 1 to %d",
                      (((validatedRow * validatedCol) * 2) - (validatedRow * validatedCol - 1
                              + (validatedCol + validatedRow))))
      );
    }

    String batChanceString = this.view.getUserInputSetting(
            "What chance do you want of a bat appearing?");
    String pitChanceString = this.view.getUserInputSetting(
            "What chance do you want of a pit appearing?");
    String arrowString = this.view.getUserInputSetting(
            "How many arrows do you want players to have");

    while (!validateDifficulty(batChanceString, pitChanceString, arrowString)) {
      batChanceString = this.view.getUserInputSetting(
              "What chance do you want of a bat appearing?");
      pitChanceString = this.view.getUserInputSetting(
              "What chance do you want of a pit appearing?");
      arrowString = this.view.getUserInputSetting("How many arrows do you want players to have");
    }

    String wrappingString = this.view.getUserInputSetting("Do you want the maze to wrap?"
            + " 1 for yes 2 for no");
    while (!wrapperHelper(wrappingString)) {
      wrappingString = this.view.getUserInputSetting("Do you want the maze to wrap?"
              + " 1 for yes 2 for no");

    }

    String playerString = this.view.getUserInputSetting("How many players? 1 or 2");
    while (!twoPlayerHelper(playerString)) {
      playerString = this.view.getUserInputSetting("How many players? 1 or 2");
    }


    String seedString = this.view.getUserInputSetting(
            String.format("The current seed is %d, if you'd like to use this input 1,"
                            + " if you'd like a new random seed input 2,"
                            + " if you'd like to put your own custom seed input 3",
                    validatedSeed)
    );

    while (!validateSeed(seedString)) {
      seedString = this.view.getUserInputSetting(
              String.format("The current seed is %d, if "
                              + "you'd like to use this input 1,"
                              + " if you'd like a new random seed input 2,"
                              + " if you'd like to put your own custom seed input 3",
                      validatedSeed)
      );
    }

    model = new HuntTheWumpus(validatedRow,
            validatedCol,
            validatedWrapper,
            validatedx,
            validatedy,
            validatedSeed,
            validatedPit,
            validatedBat,
            validatedArrow,
            validatedNumbefOfWalls,
            validatedPlayer);


    while (checkGameState()) {
      view.pushMessage(turnPlayer ? "Player 2's turn" : "Player 1's turn");
      view.pushMessage(String.format(
              "You are in cell %s%s%s",
              model.getPlayerPos(turnPlayer).toString().toLowerCase(),
              (model.getAllAdjacentContent(turnPlayer).contains(Contents.WUMPUS)
                      ? " You smell the wumpus" : ""),
              (model.getAllAdjacentContent(turnPlayer).contains(Contents.PIT)
                      ? " You feel a breeze" : "")
      ));
      String move = view.getUserInputSetting(String.format("You are %s type your possible moves "
                      + "are, %s type one of those or type shoot to change your mode",
              (validatedShoot ? "shooting" : "moving"),
              model.possibleMoves(model.getPlayerPos(turnPlayer)).toString().toLowerCase()));
      if (validatedShoot) {
        while (validatedRoom <= 0) {
          try {
            validatedRoom = Integer.parseInt(view.getUserInputSetting("How many rooms away do "
                    + "you want to shoot."));
          } catch (NumberFormatException nfe) {
            // do nothing
          }
        }
      }
      doMove(move, validatedShoot, validatedRoom);
    }


  }

  @Override
  public boolean validateSize(String colString, String rowString) {
    try {
      int col = Integer.parseInt(colString);
      int row = Integer.parseInt(rowString);

      if (col < 1 || row < 1) {
        return false;
      } else {
        validatedCol = col;
        validatedRow = row;
        return true;
      }
    } catch (NumberFormatException nfe) {
      return false;
    }
  }

  @Override
  public boolean validatePlayerPos(String playerxstring, String playerystring) {
    try {
      int x = Integer.parseInt(playerxstring);
      int y = Integer.parseInt(playerystring);
      if (x < 0 || x >= validatedCol || y < 0 || y >= validatedRow) {
        return false;
      } else {
        validatedx = x;
        validatedy = y;
        return true;
      }
    } catch (NumberFormatException nfe) {
      return false;
    }
  }

  @Override
  public boolean validateSeed(String seedString) {
    try {
      int seedInt = Integer.parseInt(seedString);
      if (seedInt == 1) {
        return true;
      } else if (seedInt == 2) {
        validatedSeed = new Random().nextInt();
        return true;
      } else if (seedInt == 3) {
        boolean temp = true;
        while (temp) {
          try {
            validatedSeed = Integer.parseInt(view.getUserInputSetting(
                    "Provide any integer to be the seed"));
            temp = false;
          } catch (NumberFormatException nfe) {
            // Do nothing we need a int for the string.
          }
        }
        return true;
      } else {
        return false;
      }
    } catch (NumberFormatException nfe) {
      return false;
    }
  }

  @Override
  public boolean validateDifficulty(String batChance, String pitChance, String arrows) {
    int batI;
    int pitI;
    int arrowI;

    try {
      batI = Integer.parseInt(batChance);
      pitI = Integer.parseInt(pitChance);
      arrowI = Integer.parseInt(arrows);
      if (batI < 0 || pitI < 0 || arrowI < 0 || batI + pitI > 100) {
        return false;
      } else {
        validatedBat = batI;
        validatedPit = pitI;
        validatedArrow = arrowI;
        return true;
      }
    } catch (NumberFormatException nfe) {
      return false;
    }
  }

  @Override
  public boolean validateNumberOfWalls(String numberOfWallsString) {
    int upperBound = ((validatedRow * validatedCol) * 2) - (validatedRow * validatedCol - 1
            + (validatedCol + validatedRow));
    try {
      int n = Integer.parseInt(numberOfWallsString);
      if (n < 1 || n > upperBound) {
        return false;
      } else {
        validatedNumbefOfWalls = n;
        return true;
      }
    } catch (NumberFormatException nfe) {
      return false;
    }
  }

  @Override
  public void doMove(String move, boolean shoot, int rooms) {
    CardinalDirections cd;
    switch (move.toLowerCase()) {
      case "shoot":
        validatedShoot = !validatedShoot;
        return;
      case "north":
        cd = CardinalDirections.NORTH;
        break;
      case "south":
        cd = CardinalDirections.SOUTH;
        break;
      case "east":
        cd = CardinalDirections.EAST;
        break;
      case "west":
        cd = CardinalDirections.WEST;
        break;
      default:
        return;
    }
    if (shoot) {
      model.shootArrow(cd, rooms, turnPlayer);
      turnPlayer = !turnPlayer;
      validatedShoot = false;
      validatedRoom = 0;
    } else if (model.possibleMoves(model.getPlayerPos(turnPlayer)).contains(cd)) {
      model.movePlayer(cd, turnPlayer);
      turnPlayer = !turnPlayer;
      validatedShoot = false;
    }
  }

  /**
   * Helper function for getting the wrapper.
   *
   * @param wrappeString the string input for the wrapper
   * @return true, if valid false if not.
   */
  private boolean wrapperHelper(String wrappeString) {
    try {
      int wrappeInt = Integer.parseInt(wrappeString);
      if (wrappeInt == 1) {
        validatedWrapper = true;
        return true;
      } else if (wrappeInt == 2) {
        validatedWrapper = false;
        return true;
      } else {
        return false;
      }
    } catch (NumberFormatException nfe) {
      return false;
    }
  }

  /**
   * Helper function for two player helper.
   *
   * @param twoPlayeString the string input for two players
   * @return true if valid, false if not
   */
  private boolean twoPlayerHelper(String twoPlayeString) {
    try {
      int wrappeInt = Integer.parseInt(twoPlayeString);
      if (wrappeInt == 1) {
        validatedPlayer = false;
        return true;
      } else if (wrappeInt == 2) {
        validatedPlayer = true;
        return true;
      } else {
        return false;
      }
    } catch (NumberFormatException nfe) {
      return false;
    }
  }

  /**
   * Checks the game state, used to decide if the game should end or not.
   *
   * @return true, if game should continue false if not.
   */
  private boolean checkGameState() {
    GameState gs = model.checkGameState(validatedPlayer);
    if (gs == GameState.FULLOSS) {
      return false;
    } else if (gs == GameState.PLAYER1LOSS) {
      turnPlayer = true;
      return true;
    } else if (gs == GameState.PLAYER2LOSS) {
      turnPlayer = false;
      return true;
    } else if (gs == GameState.PLAYER1WIN) {
      view.pushMessage("Player 1 has shot the wumpus and won!");
      turnPlayer = false;
      return false;
    } else if (gs == GameState.PLAYER2WIN) {
      view.pushMessage("Player 2 has shot the wumpus and won!");
      turnPlayer = false;
      return false;
    } else if (gs == GameState.INPROGRESS) {
      return true;
    }
    return true;
  }
}
