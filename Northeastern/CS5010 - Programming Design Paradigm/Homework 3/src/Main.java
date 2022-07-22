import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import maze.CardinalDirections;
import maze.Maze;
import maze.MazeImpl;

/**
 * Main class used to run the program.
 */
public class Main {
  private static Map<Character, CardinalDirections> move
          = new HashMap<Character, CardinalDirections>();

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
    int argIndex = 0;

    boolean perfectFlag = true;
    int n = 0;
    boolean wrappedFlag = false;
    int seed = new Random().nextInt();

    int rows = -999;
    int cols = -999;
    int goldValue = -999;
    int playerX = -999;
    int playerY = -999;
    int goalX = -999;
    int goalY = -999;

    while (argIndex < args.length) {
      if (args[argIndex].startsWith("-")) {
        if (args[argIndex].equals("-rows")) {
          try {
            rows = Integer.parseInt(args[argIndex + 1]);
            argIndex += 2;
          } catch (NumberFormatException e) {
            System.out.println("Please give an integer number following the argument -rows.");
            System.exit(1);
          }
        } else if (args[argIndex].equals("-cols")) {
          try {
            cols = Integer.parseInt(args[argIndex + 1]);
            argIndex += 2;
          } catch (NumberFormatException e) {
            System.out.println("Please give an integer number following the argument -cols.");
            System.exit(1);
          }
        } else if (args[argIndex].equals("-playerX")) {
          try {
            playerX = Integer.parseInt(args[argIndex + 1]);
            argIndex += 2;
          } catch (NumberFormatException e) {
            System.out.println("Please give an integer number following the argument -playerX.");
            System.exit(1);
          }
        } else if (args[argIndex].equals("-playerY")) {
          try {
            playerY = Integer.parseInt(args[argIndex + 1]);
            argIndex += 2;
          } catch (NumberFormatException e) {
            System.out.println("Please give an integer number following the argument -playerY.");
            System.exit(1);
          }
        } else if (args[argIndex].equals("-goalX")) {
          try {
            goalX = Integer.parseInt(args[argIndex + 1]);
            argIndex += 2;
          } catch (NumberFormatException e) {
            System.out.println("Please give an integer number following the argument -goalX.");
            System.exit(1);
          }
        } else if (args[argIndex].equals("-goalY")) {
          try {
            goalY = Integer.parseInt(args[argIndex + 1]);
            argIndex += 2;
          } catch (NumberFormatException e) {
            System.out.println("Please give an integer number following the argument -goalY.");
            System.exit(1);
          }
        } else if (args[argIndex].equals("-goldValue")) {
          try {
            goldValue = Integer.parseInt(args[argIndex + 1]);
            argIndex += 2;
          } catch (NumberFormatException e) {
            System.out.println("Please give an integer number following the argument -goldValue.");
            System.exit(1);
          }
        } else if (args[argIndex].equals("-seed")) {
          try {
            seed = Integer.parseInt(args[argIndex + 1]);
            argIndex += 2;
          } catch (NumberFormatException e) {
            System.out.println("Please give an integer number following the argument -seed.");
            System.exit(1);
          }
        } else if (args[argIndex].equals("-imperfect")) {
          try {
            n = Integer.parseInt(args[argIndex + 1]);
            perfectFlag = false;
            argIndex += 2;
          } catch (NumberFormatException e) {
            System.out.println("Please give an integer number following the argument -seed.");
            System.exit(1);
          }
        } else if (args[argIndex].equals("-wrapping")) {
          wrappedFlag = true;
          argIndex++;
        } else {
          argIndex++;
        }
      } else {
        argIndex++;
      }
    }

    for (int i : new int[]{rows, cols, goldValue, playerX, playerY, goalX, goalY}) {
      if (i == -999) {
        System.out.println("Please make sure you have all required arguments check the README for "
                + " more info");
        System.exit(1);
      }
    }

    Maze maze = new MazeImpl(rows, cols, n, perfectFlag, wrappedFlag, goldValue, playerX,
            playerY, goalX, goalY, seed);


    System.out.println("Welcome to the maze you can move around by typing the first character of "
            + "the cardinal directions (N, E, S, W).");
    while (!maze.isGoal()) {
      System.out.printf("You are currently at tile %s, your gold is %d, "
                      + "your possible moves are:%s%n", maze.getPlayerPos(), maze.getPlayerGold(),
              maze.possibleMoves());
      Character c = Character.toUpperCase(sc.next().charAt(0));
      if (move.containsKey(c)) {
        try {
          maze.movePlayer(move.get(c));
        } catch (IllegalArgumentException e) {
          System.out.println("Please select a legal move!");
        }
      } else {
        System.out.println("Please select a legal move!");
      }
    }
    System.out.printf("You Won, you completed the maze your final gold was: %d%n",
            maze.getPlayerGold());

  }
}
