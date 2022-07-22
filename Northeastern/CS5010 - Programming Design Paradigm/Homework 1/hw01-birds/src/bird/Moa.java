package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class representation of a Mao.
 */
public final class Moa extends FlightlessBird {
  /**
   * The default constructor of the Moa class.
   *
   * @param name The name of the bird.
   */
  public Moa(String name) {
    super(name, true, 0, new ArrayList<>(Arrays.asList(Food.BUDS, Food.VEGETATION)));
  }

}
