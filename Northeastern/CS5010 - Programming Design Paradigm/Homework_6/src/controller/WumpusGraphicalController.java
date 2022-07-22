package controller;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JButton;
import javax.swing.JOptionPane;
import model.CardinalDirections;
import model.GameState;
import model.HuntTheWumpus;
import model.Maze;
import view.WumpusGraphicalView;

/**
 * Implementation of Maze Controller for a GUI swing view.
 */
public final class WumpusGraphicalController implements MazeController {
  private final WumpusGraphicalView view;
  private Maze model;
  private boolean shoot = false;
  private int validatedCol;
  private int validatedRow;
  private int validatedPlayerx;
  private int validatedPlayery;
  private int validatedNumberOfWalls;
  private int validatedSeed;
  private int validatedBat;
  private int validatedPit;
  private int validatedArrow;
  private boolean turnPlayer;


  /**
   * Constructor for the controller. This controller is to be used with a Swing GUI.
   *
   * @param view the view to use with controller.
   */
  public WumpusGraphicalController(WumpusGraphicalView view) {
    this.view = view;

    this.view.newGameListener(new ListenerNewGame());
    this.view.shootButtonListener(new ShootListener());
    this.view.moveButtonListeners(new MoveListener());
    this.view.playerPosListener(new PlayerMenuListener());
    this.view.sizeListener(new SizeMenuListener());
    this.view.difficultyListener(new DifficultyMenuListener());
    this.view.wallListener(new WallMenuListener());
  }

  @Override
  public boolean validateSize(String colString, String rowString) {
    int x;
    int y;
    try {
      x = Integer.parseInt(view.getCols());
      y = Integer.parseInt(view.getRows());
      if (x > 0 && y > 0) {
        validatedCol = x;
        validatedRow = y;
        return true;
      } else {
        return false;
      }
    } catch (NumberFormatException nfe) {
      return false;
    }
  }

  @Override
  public boolean validatePlayerPos(String playerxstring, String playerystring) {
    // Assumes the row and column has been validated
    int x;
    int y;
    try {

      x = Integer.parseInt(view.getPlayerx());
      y = Integer.parseInt(view.getPlayery());
      if (x >= 0 && x < validatedCol && y >= 0 && y < validatedRow) {
        validatedPlayerx = x;
        validatedPlayery = y;
        return true;
      } else {
        return false;
      }
    } catch (NumberFormatException nfe) {
      return false;
    }
  }

  @Override
  public boolean validateSeed(String seedString) {
    try {
      validatedSeed = Integer.parseInt(seedString);
      return true;
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
      } else  {
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
    int x;
    int upperBound = (((validatedRow * validatedCol) * 2) - (validatedRow * validatedCol - 1
            + (validatedCol + validatedRow)));

    try {
      x = Integer.parseInt(numberOfWallsString);
      if (x <= 0 || x > upperBound) {
        return false;
      }
      validatedNumberOfWalls = x;
      return true;
    } catch (NumberFormatException nfe) {
      return false;
    }
  }

  @Override
  public void doMove(String move, boolean shoot, int rooms) {
    CardinalDirections direction;
    switch  (move) {
      case "north":
        direction = CardinalDirections.NORTH;
        break;
      case "south":
        direction = CardinalDirections.SOUTH;
        break;
      case "east":
        direction = CardinalDirections.EAST;
        break;
      case "west":
        direction = CardinalDirections.WEST;
        break;
      default:
        return;
    }
    if (shoot) {
      rooms = -999;
      String input = "";
      while (rooms <= 0 && input != null) {
        try {
          input = JOptionPane.showInputDialog(view, "How many rooms do you want to shoot to?");
          rooms = Integer.parseInt(input);
          model.shootArrow(direction, rooms, turnPlayer);
          view.setShootingButton(String.format("Shoot %d", model.getPlayerArrows(turnPlayer)));
        } catch (NumberFormatException nfe) {
          // do nothing
        }
      }
    } else {
      view.discoverCell(model.getAdjacent(model.getPlayerPos(turnPlayer), direction));
      model.movePlayer(direction, turnPlayer);
      view.discoverCell(model.getPlayerPos(turnPlayer));
    }
    view.updatedMazeGrid();
    shoot = false;
    view.setStatus("Moving");
    checkGameState();

  }

  /**
   * Helper Function to determine the end of game state, passes along to the view that the game has
   * ended in push method.
   */
  private void checkGameState() {
    GameState gs = model.checkGameState(view.getTwoPlayer());
    if (gs == GameState.FULLOSS) {
      view.setEndGame("All players have lost the game is over.");
    } else if (gs == GameState.PLAYER1LOSS) {
      turnPlayer = true;
    } else if (gs == GameState.PLAYER2LOSS) {
      turnPlayer = false;
    } else if (gs == GameState.PLAYER1WIN) {
      view.setEndGame("Player 1 has shot the wumpus and won!");
      turnPlayer = false;
    } else  if (gs == GameState.PLAYER2WIN) {
      view.setEndGame("Player 2 has shot the wumpus and won!");
      turnPlayer = false;
    } else if (gs == GameState.INPROGRESS) {
      turnPlayer = !turnPlayer;
    }
    view.setTurnPlayer(turnPlayer);
    view.updatedMazeGrid();
  }

  /**
   * Private class for the size menu item to add the listener.
   */
  private class SizeMenuListener implements ActionListener {

    @Override
    public void actionPerformed(ActionEvent e) {
      String valid = null;
      view.getBoardSize(null);
      while (!validateSize(view.getCols(), view.getRows())) {
        if (view.getBoardSize(valid)) {
          valid = "Values should be integers which are greater than 0";
        } else {
          break;
        }
      }
    }
  }

  /**
   * Action Listener class for the player position menu item.
   */
  private class PlayerMenuListener implements ActionListener {

    @Override
    public void actionPerformed(ActionEvent e) {
      String valid = null;

      if (!validateSize(view.getCols(), view.getRows())) {
        view.setStatus("Please provide a valid size of board!");
        return;
      }
      view.getPlayerPos(null);
      while (!validatePlayerPos(view.getPlayerx(), view.getPlayery())) {
        if (view.getPlayerPos(valid)) {
          valid = "Please provide integer values which are within the boards size!";
        } else {
          break;
        }
      }
    }
  }


  /**
   * Action Listener class for the number of walls the maze has.
   */
  private class WallMenuListener implements ActionListener {

    @Override
    public void actionPerformed(ActionEvent e) {
      String valid = null;

      if (!validateSize(view.getCols(), view.getRows())) {
        view.setStatus("Please provide a valid size of board!");
        return;
      }
      view.getNumberOfWallInput(null);
      while (!validateNumberOfWalls(view.getNumberOfWalls())) {
        if (view.getNumberOfWallInput(valid)) {
          valid = "Please provide integer values between 1 and "
                  + (((validatedRow * validatedCol) * 2) - (validatedRow * validatedCol - 1
                  + (validatedCol + validatedRow)));
        } else {
          break;
        }
      }
    }

  }

  /**
   * Action Listener for the DifficultyMenuItem.
   */
  private class DifficultyMenuListener implements ActionListener {

    @Override
    public void actionPerformed(ActionEvent e) {
      String valid = null;
      view.getDifficultyInput(null);
      while (!validateDifficulty(view.getBatChance(), view.getPitChance(), view.getArrows())) {
        if (view.getDifficultyInput(valid)) {
          valid = "All values should be greater than 0 and chance of getting a pit or a bat should"
                  + " not add up to more than 100.";
        } else {
          break;
        }
      }
    }
  }

  /**
   * Action Listener for the new game menu item.
   */
  private class ListenerNewGame implements ActionListener {

    @Override
    public void actionPerformed(ActionEvent e) {
      view.selectSeed(null);
      while (!validateSeed(view.getSeed())) {
        view.selectSeed("Please set a seed that is an integer");
      }

      try {
        model = new HuntTheWumpus(validatedRow,
                validatedCol,
                view.wrappingChecked(),
                validatedPlayerx,
                validatedPlayery,
                validatedSeed,
                validatedPit,
                validatedBat,
                validatedArrow,
                validatedNumberOfWalls,
                view.getTwoPlayer());
        turnPlayer = false;
        view.setTurnPlayer(false);
        view.setMaze(model);
        view.setGameStarted();
        view.setMazeLabels();
        view.discoverCell(model.getPlayerPos(turnPlayer));
        view.updatedMazeGrid();
        view.setShootingButton("Shoot: " + model.getPlayerArrows(turnPlayer));
        view.setStatus("Moving");
      } catch (NumberFormatException nfe) {
        view.setStatus("Please enter the settings through the menu!");
      } catch (IllegalArgumentException iae) {
        view.setStatus("Please make sure all your setting values are legal values");
      }
    }
  }

  /**
   * ActionListener for the shoot button.
   */
  private class ShootListener implements ActionListener {
    @Override
    public void actionPerformed(ActionEvent e) {
      shoot = !shoot;
      view.setStatus(shoot ? "Shooting" : "Moving");
    }

  }

  /**
   *  ActionListener for the move buttons.
   */
  private class MoveListener implements ActionListener {

    @Override
    public void actionPerformed(ActionEvent e) {
      String buttonString = (((JButton) e.getSource()).getText().toLowerCase());
      doMove(buttonString, shoot, 0);
    }

  }
}
