import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import maze.CardinalDirections;
import maze.Contents;
import maze.WumpusController;

/**
 * Main class used to run the program.
 */
public class Main {
  private static final Map<Character, CardinalDirections> move
          = new HashMap<>();

  /**
   * Main class this is called when the program is ran from the command line.
   *
   * @param args The arguments to pass to the program/
   */
  public static void main(String[] args) {

    move.put('N', CardinalDirections.NORTH);
    move.put('W', CardinalDirections.WEST);
    move.put('E', CardinalDirections.EAST);
    move.put('S', CardinalDirections.SOUTH);
    Scanner sc = new Scanner(System.in);

    boolean perfectFlag = true;
    boolean wrappedFlag = false;
    int seed = new Random().nextInt();

    int rows;
    int cols;
    int arrows;
    int playerX;
    int playerY;
    int pitChance;
    int batChance;
    int rooms = 1;

    System.out.println("Welcome to wumpus world, lets configure your game.");
    System.out.println("First, lets decided on how big the maze is. How many rows are there:");
    rows = sc.nextInt();
    while (rows < 1) {
      System.out.println("The number of rows must be greater than 0");
      rows = sc.nextInt();
    }
    System.out.println("Next, how many columns are there:");
    cols = sc.nextInt();
    while (cols < 1) {
      cols = sc.nextInt();
      System.out.println("The number of columns must be greater than 0");
    }
    System.out.println("How many arrows should the player have");
    arrows = sc.nextInt();
    while (arrows < 1) {
      System.out.println("Must have more than 0 arrows to start");
      arrows = sc.nextInt();
    }
    System.out.println("Where would you like to start?");
    System.out.println("X(which column)?");
    playerX = sc.nextInt();
    while (0 > playerX || playerX > cols) {
      System.out.println("Players X pos has to be between 1 and number of cols");
      playerX = sc.nextInt();
    }
    System.out.println("Y(which row)?");
    playerY = sc.nextInt();
    while (0 > playerY || playerY > rows) {
      System.out.println("Players Y pos has to be between 1 and number of rows");
      playerY = sc.nextInt();
    }
    System.out.println("What chance do you want a room to have a pit");
    pitChance = sc.nextInt();
    while (pitChance < 0 || pitChance > 50) {
      System.out.println("The chance for a pit must be between 1 and 50");
      pitChance = sc.nextInt();
    }
    System.out.println("What chance do you want a room to have a bat");
    batChance = sc.nextInt();
    while (batChance < 0 || batChance > 50) {
      System.out.println("The chance for a bat must be between 1 and 50");
      batChance = sc.nextInt();
    }
    System.out.println("Do you want the maze to be perfect y/n?");
    char perfectBool = ' ';
    while (perfectBool != 'y' && perfectBool != 'n') {
      perfectBool = sc.next().toLowerCase().charAt(0);
      perfectFlag = perfectBool == 'y';
    }
    System.out.println("Do you want the maze to be wrapping? y/n?");
    char wrappingBool = ' ';
    while (wrappingBool != 'y' && wrappingBool != 'n') {
      wrappingBool = sc.next().toLowerCase().charAt(0);
      wrappedFlag = wrappingBool == 'y';
    }
    // create the controller
    WumpusController wc = new WumpusController(rows, cols, perfectFlag, wrappedFlag, playerX,
            playerY, seed, pitChance, batChance);

    char input;

    while (true) {
      System.out.printf("You are at position %d, %d, would you like to move(m) or shoot(a)?%n",
              wc.getPlayerPos().getxCoordinates(), wc.getPlayerPos().getyCoordinates());
      ArrayList<Contents> cs = wc.getAdjacent();
      System.out.printf("%s%s\n",
              (cs.contains(Contents.PIT)) ? "You feel a breeze\n" : "",
              (cs.contains(Contents.WUMPUS)) ? "You smell the wumpus\n" : "");

      input = sc.next().toLowerCase().charAt(0);
      while (input != 'm' && input != 'a') {
        System.out.println("Select a legal move!");
        input = sc.next().toLowerCase().charAt(0);
      }
      char method;
      if (input == 'm') {
        System.out.printf("Where would you like to move?\n You're legal moves are: %s%n",
                wc.getPlayerMoves().toString());
        method = 'm';
      } else {
        System.out.printf("Where would you like to Shoot?\n You're legal moves are: %s%n",
                wc.getPlayerMoves().toString());
        method = 's';
        System.out.printf("How many rooms to shoot, between %d and %d\n", 1, Math.max(rows, cols));
        rooms = sc.nextInt();
        while (rooms < 1 ||  rooms > Math.max(rows, cols)) {
          System.out.println("Choose a valid number.");
          rooms = sc.nextInt();
        }
      }
      input = sc.next().toUpperCase().charAt(0);
      while (!(move.containsKey(input)) && !(wc.getPlayerMoves().contains(move.get(input)))) {
        input = sc.next().toUpperCase().charAt(0);
        System.out.println("Select a valid direction (N, E, S, W).");
      }
      if (method == 'm') {
        wc.movePlayer(move.get(input));
        Contents content = wc.getContent();
        switch (content) {
          case WUMPUS:
            System.out.println("Chomp, chomp, chomp, thanks for feeding the Wumpus!");
            System.out.println("Better luck next time");
            System.exit(0);
            break;
          case PIT:
            System.out.println("You fall down a pit and die!");
            System.out.println("Better luck next time");
            System.exit(0);
            break;
          case BATS:
            boolean bat = wc.bat();
            if (bat) {
              System.out.println("Snatch -- you are grabbed by superbats and ...");
            } else {
              System.out.println("Whoa -- you successfully duck superbats that try to grab you");
            }
            break;
          default:
            break;
        }
      } else {
        if (wc.shoot(move.get(input), rooms)) {
          System.out.println("Hee hee hee, you got the wumpus!\n"
                  + "Next time you won't be so lucky");
          break;
        } else {
          if ( wc.emptyQuiver() ) {
            System.out.println("You ran out of arrows! \n Best of Luck Next time!");
            break;
          }
        }
      }
    }
  }
}
