import controller.WumpusConsoleController;
import controller.WumpusGraphicalController;
import view.WumpusConsoleView;
import view.WumpusGraphicalView;

/**
 * Main Driver.
 */
public class MainDriver {
  /**
   * Main driver method.

   * @param args arguments
   */
  public static void main(String[] args) {
    if (args[0].equals("--gui")) {
      WumpusGraphicalView view = new WumpusGraphicalView();

      new WumpusGraphicalController(view);
    } else if (args[0].equals("--text")) {
      WumpusConsoleView view = new WumpusConsoleView(System.in);

      new WumpusConsoleController(view);
    } else {
      System.out.println("Please make sure you have the correct argument either --gui or --text");
    }

  }
}
