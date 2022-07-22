package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class representation of a gray parrot.
 */
public final class GrayParrot extends Parrot {
  /**
   * default constructor for the gray parrot.
   *
   * @param name           The name of the bird
   * @param vocabSize      The size of the bird's vocab.
   * @param favoritePhrase The bird's favorite phrase
   */
  public GrayParrot(String name, int vocabSize, String favoritePhrase) {
    super(name, false, 2, new ArrayList<>(Arrays.asList(Food.SEEDS, Food.VEGETATION)),
            vocabSize, favoritePhrase);
  }

}
