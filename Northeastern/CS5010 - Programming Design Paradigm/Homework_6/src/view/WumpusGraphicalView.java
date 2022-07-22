package view;

import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import javax.imageio.ImageIO;
import javax.swing.AbstractAction;
import javax.swing.ButtonGroup;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JTextField;
import javax.swing.KeyStroke;
import model.CardinalDirections;
import model.Contents;
import model.Coordinates;
import model.Maze;

/**
 * Wumpus Swing graphical view.
 */
public class WumpusGraphicalView extends JFrame {
  private final JPanel[][] mazeGridPanels = new JPanel[3][3];
  private final JPanel mazePanel;

  private JButton shootButton;
  private JButton northButton;
  private JButton eastButton;
  private JButton southButton;
  private JButton westButton;

  private JLabel[][] gridLabels;
  private JLabel statusLabel;
  private JLabel turnLabel;

  private JMenuItem newGameItem;
  private JMenuItem setPlayerCoordinates;
  private JMenuItem setSizeOfGame;
  private JMenuItem setNumberOfWalls;
  private JMenuItem setDifficulty;
  private JCheckBoxMenuItem wrapping;
  private JCheckBoxMenuItem twoPlayer;

  private String rows;
  private String cols;
  private String playerx;
  private String playery;
  private String numberOfWalls;
  private String batChance;
  private String pitChance;
  private String arrows;
  private String seed = Integer.toString(new Random().nextInt());
  private boolean randomSeed;

  private boolean gameStarted;
  private boolean turnPlayer;
  private Maze maze;

  /**
   * Graphical swing view constructor.
   */
  public WumpusGraphicalView() {
    this.gameStarted = false;

    GridLayout mainGrid = new GridLayout(3, 0);
    JPanel mainPanel = new JPanel();
    mainPanel.setLayout(mainGrid);

    GridLayout mazeGrid = new GridLayout(3, 3);
    mazeGrid.setVgap(0);
    mazeGrid.setHgap(0);
    this.mazePanel = new JPanel();
    this.mazePanel.setSize(323, 323);
    this.mazePanel.setLayout(mazeGrid);

    JLabel defaultTile = new JLabel();
    defaultTile.setIcon(new ImageIcon("src/res/black.png"));

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        JPanel tmp = new JPanel();
        tmp.setSize(64, 64);
        mazeGridPanels[j][i] = tmp;
        this.mazePanel.add(mazeGridPanels[j][i]);
      }
    }

    mazeGridPanels[1][1].add(defaultTile);

    mainPanel.add(this.mazePanel);

    JPanel buttonPanel = new JPanel();
    GridLayout buttonGrid = new GridLayout(3, 3);
    buttonPanel.setLayout(buttonGrid);
    JPanel[][] buttonGridPanels = new JPanel[3][3];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        buttonGridPanels[j][i] = new JPanel();
        buttonPanel.add(buttonGridPanels[j][i]);
      }
    }

    this.northButton = new JButton("North");
    this.northButton.setEnabled(gameStarted);
    buttonGridPanels[1][0].add(this.northButton);

    this.eastButton = new JButton("East");
    this.eastButton.setEnabled(gameStarted);
    buttonGridPanels[2][1].add(this.eastButton);

    this.southButton = new JButton("South");
    this.southButton.setEnabled(gameStarted);
    buttonGridPanels[1][2].add(this.southButton);

    this.westButton = new JButton("West");
    this.westButton.setEnabled(gameStarted);
    buttonGridPanels[0][1].add(this.westButton);

    this.shootButton = new JButton("Shoot");
    this.shootButton.setEnabled(gameStarted);
    buttonGridPanels[1][1].add(this.shootButton);

    mainPanel.add(buttonPanel);

    JPanel statusPanel = new JPanel();
    this.turnLabel = new JLabel();
    statusPanel.add(this.turnLabel);
    this.statusLabel = new JLabel("Welcome!");
    statusPanel.add(statusLabel);

    mainPanel.add(statusPanel);

    JMenuBar menuBar = new JMenuBar();

    JMenu fileMenu = new JMenu("File");
    this.newGameItem = new JMenuItem("New Game");
    fileMenu.add(this.newGameItem);

    menuBar.add(fileMenu);

    JMenu settingMenu = new JMenu("Settings");
    this.setSizeOfGame = new JMenuItem("Size of Board");
    settingMenu.add(this.setSizeOfGame);
    this.setPlayerCoordinates = new JMenuItem("Set Player Coordinates");
    settingMenu.add(this.setPlayerCoordinates);
    this.setNumberOfWalls = new JMenuItem("Set Number of Walls");
    settingMenu.add(setNumberOfWalls);
    this.setDifficulty = new JMenuItem("Set Difficulty");
    settingMenu.add(this.setDifficulty);
    this.wrapping = new JCheckBoxMenuItem("Wrapping");
    settingMenu.add(this.wrapping);
    this.twoPlayer = new JCheckBoxMenuItem("Two Players");
    settingMenu.add(this.twoPlayer);

    menuBar.add(settingMenu);

    this.setJMenuBar(menuBar);

    NorthAction northAction = new NorthAction();
    SouthAction southAction = new SouthAction();
    EastAction eastAction = new EastAction();
    WestAction westAction = new WestAction();
    SpaceAction spaceAction = new SpaceAction();

    buttonPanel.getInputMap().put(KeyStroke.getKeyStroke('w'), "northAction");
    buttonPanel.getActionMap().put("northAction", northAction);
    buttonPanel.getInputMap().put(KeyStroke.getKeyStroke('s'), "southAction");
    buttonPanel.getActionMap().put("southAction", southAction);
    buttonPanel.getInputMap().put(KeyStroke.getKeyStroke('d'), "eastAction");
    buttonPanel.getActionMap().put("eastAction", eastAction);
    buttonPanel.getInputMap().put(KeyStroke.getKeyStroke('a'), "westAction");
    buttonPanel.getActionMap().put("westAction", westAction);
    buttonPanel.getInputMap().put(KeyStroke.getKeyStroke(' '), "spaceAction");
    buttonPanel.getActionMap().put("spaceAction", spaceAction);

    this.add(mainPanel);
    this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    this.pack();
    this.setVisible(true);
  }

  /**
   * Updates the maze grid.
   */
  public void updatedMazeGrid() {
    if (gameStarted) {
      int playerX = maze.getPlayerPos(turnPlayer).getXcoordinates();
      int playerY = maze.getPlayerPos(turnPlayer).getYcoordinates();
      for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {

          if (testCoord((playerX - 1 + x), (playerY - 1 + y))) {
            mazeGridPanels[x][y].removeAll();
            mazeGridPanels[x][y].add(gridLabels[playerX - 1 + x][playerY - 1 + y]);
          } else {
            mazeGridPanels[x][y].removeAll();
            mazeGridPanels[x][y].add(new JLabel(new ImageIcon("src/res/black.png")));
          }
        }
      }
      try {
        mazeGridPanels[1][1].removeAll();
        ImageIcon icon = (ImageIcon) gridLabels[playerX][playerY].getIcon();
        JLabel playerLabel;
        if (turnPlayer) {
          playerLabel = new JLabel(new ImageIcon(combineImages(icon.getImage(),
                  ImageIO.read(new File("src/res/player2.png")))));
        } else {
          playerLabel = new JLabel(new ImageIcon(combineImages(icon.getImage(),
                  ImageIO.read(new File("src/res/player.png")))));
        }
        mazeGridPanels[1][1].add(playerLabel);

      } catch (IOException io) {
        // do nothing, this grabs pre-defined images
      }
    }
    this.mazePanel.repaint();
    this.revalidate();
  }

  /**
   * Set ths listener for hte new game item.
   *
   * @param listenForNewGameItem the listener
   */
  public void newGameListener(ActionListener listenForNewGameItem) {

    newGameItem.addActionListener(listenForNewGameItem);

  }

  /**
   * Returns if the the two player setting is checked.
   *
   * @return true if the two player is selected else false
   */
  public boolean getTwoPlayer() {
    return this.twoPlayer.isSelected();
  }

  /**
   * Sets the model which is a maze.
   *
   * @param maze the maze model to set.
   */
  public void setMaze(Maze maze) {
    this.maze = maze;
  }

  /**
   * This is done to set the maze labels up at the begining.
   */
  public void setMazeLabels() {
    if (gameStarted) {
      int cols = maze.getMaxPosition().getXcoordinates() + 1;
      int rows = maze.getMaxPosition().getYcoordinates() + 1;
      this.gridLabels = new JLabel[cols][rows];
      for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
          this.gridLabels[x][y] = new JLabel(new ImageIcon("src/res/unkown.png"));
        }
      }
    }
  }

  /**
   * Sets all the enabled flags for the start of the game.
   */
  public void setGameStarted() {
    this.gameStarted = true;
    this.shootButton.setEnabled(true);
    this.northButton.setEnabled(true);
    this.eastButton.setEnabled(true);
    this.southButton.setEnabled(true);
    this.westButton.setEnabled(true);
    this.twoPlayer.setEnabled(false);
  }

  /**
   * Sets all the flags to their correct state for the end game also sets the end game message.
   *
   * @param s the end game message
   */
  public void setEndGame(String s) {
    this.gameStarted = false;
    this.shootButton.setEnabled(false);
    this.northButton.setEnabled(false);
    this.eastButton.setEnabled(false);
    this.southButton.setEnabled(false);
    this.westButton.setEnabled(false);
    this.twoPlayer.setEnabled(true);
    this.statusLabel.setText(s);
  }

  /** Test Coordinates for drawing, the board.
   *
   * @param x x coordinate
   * @param y y coordinate
   * @return true if on  board, else false
   */
  private boolean testCoord(int x, int y) {
    if (x < 0 || x > maze.getMaxPosition().getXcoordinates()) {
      return false;
    } else if (y < 0 || y > maze.getMaxPosition().getYcoordinates()) {
      return false;
    }
    return true;
  }

  /**
   * Combines 2 images, layering them on top of each other.
   *
   * @param a the base image
   * @param b the image to layer on top
   * @return the combined images
   */
  private Image combineImages(Image a, Image b) {
    BufferedImage combinedImage = new BufferedImage(a.getWidth(null), a.getHeight(null),
            BufferedImage.TYPE_INT_ARGB);

    int offsetX = (a.getWidth(null) - b.getWidth(null)) / 2;
    int offsetY = (a.getHeight(null) - b.getHeight(null)) / 2;


    Graphics g = combinedImage.getGraphics();
    g.drawImage(a, 0, 0, null);
    g.drawImage(b, offsetX, offsetY, null);

    return combinedImage;
  }

  /**
   * Decides what tile to return for the tile give.
   *
   * @param possibleMoves The possibles moves the current cell
   * @param content the content of the current cell
   * @param adj the adjacent content that matters.
   * @return the image of the title
   */
  private Image decideTile(ArrayList<CardinalDirections> possibleMoves, Contents content,
                           ArrayList<Contents> adj) {
    StringBuilder sb = new StringBuilder();

    if (possibleMoves.contains(CardinalDirections.NORTH)) {
      sb.append("n");
    }
    if (possibleMoves.contains(CardinalDirections.SOUTH)) {
      sb.append("s");
    }
    if (possibleMoves.contains(CardinalDirections.EAST)) {
      sb.append("e");
    }
    if (possibleMoves.contains(CardinalDirections.WEST)) {
      sb.append("w");
    }
    Image room = null;
    try {
      room = ImageIO.read(new File("src/res/" + sb.toString().toUpperCase() + ".png"));
    } catch (IOException io) {
      // Do nothing for now.
    }

    try {
      Image contentImage;
      switch (content) {
        case PIT:
          contentImage = ImageIO.read(new File("src/res/pit.png"));
          break;
        case WUMPUS:
          contentImage = ImageIO.read(new File("src/res/wumpus.png"));
          break;
        case BATS:
          contentImage = ImageIO.read(new File("src/res/bats.png"));
          break;
        default:
          contentImage = ImageIO.read(new File("src/res/blank.png"));
      }

      room = combineImages(room, contentImage);
    } catch (IOException io) {
      // DO NOTHING
    }


    if (adj.contains(Contents.WUMPUS)) {
      try {
        Image pit = ImageIO.read(new File("src/res/stench.png"));
        room = combineImages(room, pit);
      } catch (IOException io) {
        //Do nothing loading predefined image
      }
    }
    if (adj.contains(Contents.PIT)) {
      try {
        Image pit = ImageIO.read(new File("src/res/breeze.png"));
        room = combineImages(room, pit);
      } catch (IOException io) {
        //Do nothing loading predefined image
      }
    }

    return room;
  }

  /**
   * Discovers a cell so it can be displayed on the gui.
   *
   * @param coord the coordinate which the cell is at
   */
  public void discoverCell(Coordinates coord) {
    Image cell = decideTile(maze.possibleMoves(coord), maze.getContent(coord),
            maze.getAllAdjacentContent(turnPlayer));
    gridLabels[coord.getXcoordinates()][coord.getYcoordinates()] = new JLabel(new ImageIcon(cell));
  }

  /**
   * Sets the action listener for the shoot button.
   *
   * @param shootListener the action listener.
   */
  public void shootButtonListener(ActionListener shootListener) {
    shootButton.addActionListener(shootListener);
  }

  /**
   * Sets the action listener for the move buttons.
   *
   * @param moveListener the action listener
   */
  public void moveButtonListeners(ActionListener moveListener) {
    northButton.addActionListener(moveListener);
    eastButton.addActionListener(moveListener);
    southButton.addActionListener(moveListener);
    westButton.addActionListener(moveListener);
  }

  /**
   * Set the text of the shoot button.
   *
   * @param s the string to set the text too.
   */
  public void setShootingButton(String s) {
    this.shootButton.setText(s);
  }

  /**
   * Sets the status message.
   *
   * @param s the message to set the status too.
   */
  public void setStatus(String s) {
    this.statusLabel.setText(s);
  }

  /**
   * Returns whether the wrapped setting is checked.
   *
   * @return true if wrapped menu item is checked
   */
  public boolean wrappingChecked() {
    return this.wrapping.isSelected();
  }

  /**
   * Sets the listener for the player position menu item.
   *
   * @param e the listener
   */
  public void playerPosListener(ActionListener e) {
    this.setPlayerCoordinates.addActionListener(e);
  }

  /**
   * Sets the listener for the board size menu item.
   *
   * @param e the listener
   */
  public void sizeListener(ActionListener e) {
    this.setSizeOfGame.addActionListener(e);
  }

  /**
   * sets the listener for the difficulty menu item.
   *
   * @param e the listener
   */
  public void difficultyListener(ActionListener e) {
    this.setDifficulty.addActionListener(e);
  }

  /**
   * Sets the listener for the number of walls menu item.
   *
   * @param e the listener
   */
  public void wallListener(ActionListener e) {
    this.setNumberOfWalls.addActionListener(e);
  }

  /**
   * Grabs the user input  for board size and sets it.
   *
   * @param validation validation string to explain if there was a validation problem with the
   *                   data.
   * @return true if value was set false if the input was canceled.
   */
  public boolean getBoardSize(String validation) {
    if (validation == null) {
      validation = "Provide the size of the board.\n";
    }

    JTextField colsF = new JTextField(this.cols);
    JTextField rowsF = new JTextField(this.rows);
    Object[] message = {
      validation,
      "Columns", colsF,
      "Rows:", rowsF
    };
    int option = JOptionPane.showConfirmDialog(this, message,
            "Player Position", JOptionPane.OK_CANCEL_OPTION);
    if (option == JOptionPane.OK_OPTION) {
      this.rows = rowsF.getText();
      this.cols = colsF.getText();
      return true;
    } else {
      return false;
    }
  }

  /**
   * Grabs the user input for player position and sets it.
   *
   * @param validation validation string to explain if there was a validation problem with the
   *                   data.
   * @return true if value was set false if the input was canceled.
   */
  public boolean getPlayerPos(String validation) {
    if (validation == null) {
      validation = "Please provide the player(s) starting position.\n";
    }
    JTextField playerX = new JTextField(this.playerx);
    JTextField playerY = new JTextField(this.playery);
    Object[] message = {
      validation,
      "PlayerX:", playerX,
      "PlayerY:", playerY
    };

    int option = JOptionPane.showConfirmDialog(this, message,
            "Player Position", JOptionPane.OK_CANCEL_OPTION);
    if (option == JOptionPane.OK_OPTION) {
      this.playerx = playerX.getText();
      this.playery = playerY.getText();
      return true;
    } else {
      return false;
    }
  }

  /**
   * Grabs the user input  for the number of walls and sets it.
   *
   * @param validation validation string to explain if there was a validation problem with the
   *                   data.
   * @return true if value was set false if the input was canceled.
   */
  public boolean getNumberOfWallInput(String validation) {
    if (validation == null) {
      int rowsTemp = 0;
      int colsTemp = 0;
      try {
        rowsTemp = Integer.parseInt(this.rows);
        colsTemp = Integer.parseInt(this.cols);
      } catch (NumberFormatException nfe) {
        // do nothing this should not be possible to reach !
      }
      validation = String.format("please provide the number of walls,"
                      + " this must be between 1 and %d",
              ((rowsTemp * colsTemp) * 2) - (rowsTemp * colsTemp - 1 + (colsTemp + rowsTemp)));
    }

    JTextField wallF = new JTextField(this.numberOfWalls);
    Object[] message = {
      validation, wallF
    };

    int option = JOptionPane.showConfirmDialog(this, message, "Player Position",
            JOptionPane.OK_CANCEL_OPTION);
    if (option == JOptionPane.OK_OPTION) {
      numberOfWalls = wallF.getText();
      return true;
    } else {
      return false;
    }
  }

  /**
   * Grabs the value of playerx.
   *
   * @return the value of playerx
   */
  public String getPlayerx() {
    return playerx;
  }

  /**
   * Gets the playery value.
   *
   * @return the value of playery
   */
  public String getPlayery() {
    return playery;
  }

  /**
   * Grabs the row value.
   *
   * @return the row value
   */
  public String getRows() {
    return rows;
  }

  /**
   * Grabs the col value.
   *
   * @return the col value
   */
  public String getCols() {
    return cols;
  }

  /**
   * Grabs the number of walls value.
   *
   * @return the number of walls value
   */
  public String getNumberOfWalls() {
    return numberOfWalls;
  }

  /**
   * Grabs the user input  for the difficulty and sets it.
   *
   * @param validation validation string to explain if there was a validation problem with the
   *                   data.
   * @return true if value was set false if the input was canceled.
   */
  public boolean getDifficultyInput(String validation) {
    if (validation == null) {
      validation = "Please enter in the chance of a bat appearing, "
              + "a pit appearing and how many arrows a player should have.";
    }
    JTextField batF = new JTextField(batChance);
    JTextField pitF = new JTextField(pitChance);
    JTextField arrowF = new JTextField(arrows);
    Object[] message = {
      validation,
      "Chance of Bats (%):", batF,
      "Chance of pits (%):", pitF,
      "Number of arrows:", arrowF
    };

    int option = JOptionPane.showConfirmDialog(this, message, "Player Position",
            JOptionPane.OK_CANCEL_OPTION);
    if (option == JOptionPane.OK_OPTION) {
      batChance = batF.getText();
      pitChance = pitF.getText();
      arrows = arrowF.getText();
      return true;
    } else {
      return false;
    }
  }

  /**
   * Grabs the chance of a bat value.
   *
   * @return the chance of a bat value
   */
  public String getBatChance() {
    return batChance;
  }

  /**
   * Grabs the chat of a pit value.
   *
   * @return the chat of a pit value
   */
  public String getPitChance() {
    return pitChance;
  }

  /**
   * Grabs the arrows value.
   *
   * @return the arrows value
   */
  public String getArrows() {
    return arrows;
  }

  /**
   * Grabs the user input  for the seed and sets it.
   *
   * @param validation validation string to explain if there was a validation problem with the
   *                   data.
   * @return true if value was set false if the input was canceled.
   */
  public boolean selectSeed(String validation) {
    if (validation == null) {
      validation = "Please select your seed";
    }
    ButtonGroup bg = new ButtonGroup();
    JTextField seedF = new JTextField(this.seed);
    seedF.setEnabled(false);
    JRadioButton rb1 = new JRadioButton("Current Seed");
    rb1.setSelected(true);
    JRadioButton rb2 = new JRadioButton("New Random Seed");
    JRadioButton rb3 = new JRadioButton("Custom Seed");
    bg.add(rb1);
    bg.add(rb2);
    bg.add(rb3);

    rb1.addActionListener(e -> {
      seedF.setEnabled(false);
      randomSeed = false;
    }
    );

    rb2.addActionListener(e -> {
      seedF.setEnabled(false);
      randomSeed = true;
    }
    );
    rb3.addActionListener(e -> {
      seedF.setEnabled(true);
      randomSeed = false;
    });

    Object[] message = {
      validation,
      rb1,
      rb2,
      rb3,
      seedF
    };

    JOptionPane.showConfirmDialog(this, message, "Set Seed",
            JOptionPane.DEFAULT_OPTION, JOptionPane.PLAIN_MESSAGE);
    if (randomSeed) {
      seed = Integer.toString(new Random().nextInt());
    } else {
      seed = seedF.getText();
    }
    return randomSeed;
  }

  /**
   * Grabs the seed value.
   *
   * @return the seed value
   */
  public String getSeed() {
    return seed;
  }

  /** Sets the turn player. True for player 2 false for player 1.
   *
   * @param turnPlayer the turn player true for player 2 false for player 1
   */
  public void setTurnPlayer(boolean turnPlayer) {
    this.turnPlayer = turnPlayer;
    this.turnLabel.setText(turnPlayer ? "Player 2" : "Player 1");
  }

  /**
   * Action to allow for keyboard input to be used for the north button.
   */
  private class NorthAction extends AbstractAction {

    @Override
    public void actionPerformed(ActionEvent e) {
      northButton.doClick();

    }
  }

  /**
   * Action to allow for keyboard input to be used for the south button.
   */
  private class SouthAction extends AbstractAction {

    @Override
    public void actionPerformed(ActionEvent e) {
      southButton.doClick();

    }
  }

  /**
   * Action to allow for keyboard input to be used for the east button.
   */
  private class EastAction extends AbstractAction {

    @Override
    public void actionPerformed(ActionEvent e) {
      eastButton.doClick();

    }
  }

  /**
   * Action to allow for keyboard input to be used for the west button.
   */
  private class WestAction extends AbstractAction {


    @Override
    public void actionPerformed(ActionEvent e) {
      westButton.doClick();
    }
  }

  /**
   * Action to allow for keyboard input to be used for the shoot button.
   */
  private class SpaceAction extends AbstractAction {

    @Override
    public void actionPerformed(ActionEvent e) {
      shootButton.doClick();
    }
  }
}