package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class representation of an osprey.
 */
public final class Osprey extends BirdOfPray {
  /**
   * The default constructor of the Osprey class.
   *
   * @param name The name of the bird.
   */
  public Osprey(String name) {
    super(name, false, 2, new ArrayList<>(Arrays.asList(Food.SMALL_MAMMALS, Food.FISH)));
  }

}
