package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class representation of an Emu.
 */
public final class Emu extends FlightlessBird {
  /**
   * The default constructor of the Emu class.
   *
   * @param name The name of the bird.
   */
  public Emu(String name) {
    super(name, false, 2, new ArrayList<>(Arrays.asList(Food.SEEDS, Food.VEGETATION)),
            new ArrayList<>(Arrays.asList("Vestigial wings", "Sharp claws")));
  }

}
