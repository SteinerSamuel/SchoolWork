package view;

import java.io.InputStream;
import java.util.Scanner;

/**
 * A console view which takes strings and outputs them to console and takes strings from an input
 * Stream  and sends them to the contreoller.
 */
public class WumpusConsoleView {
  InputStream in;

  /**
   * Constructor for the class.
   *
   * @param in input stream to get input from.
   */
  public WumpusConsoleView(InputStream in) {
    this.in = in;
  }

  /**
   * Grabs a string from the input stream provided.
   *
   * @param input A string which explains what the input is.
   * @return the string response from the input stream
   */
  public String getUserInputSetting(String input) {
    Scanner scanner = new Scanner(this.in);
    System.out.println(input);
    return scanner.nextLine();
  }

  /**
   * Push a message to the console.
   *
   * @param message the message which is pushed to the console
   */
  public void pushMessage(String message) {
    System.out.println(message);
  }
}
