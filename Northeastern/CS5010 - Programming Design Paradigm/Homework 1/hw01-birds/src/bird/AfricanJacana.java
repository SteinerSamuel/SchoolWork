package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class representation of a african jacana.
 */
public final class AfricanJacana extends ShoreBird {
  /**
   * default constructor for the african jacana.
   *
   * @param name The name of the bird.
   */
  public AfricanJacana(String name) {
    super(name, false, 2, new ArrayList<>(Arrays.asList(Food.SEEDS, Food.VEGETATION)), "Lake");
  }

}
