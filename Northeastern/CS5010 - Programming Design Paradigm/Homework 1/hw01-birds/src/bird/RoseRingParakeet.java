package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class representation of a rose ring parakeet.
 */
public final class RoseRingParakeet extends Parrot {
  /**
   * default constructor for the Rose ring parakeet.
   *
   * @param name           The name of the bird.
   * @param vocabSize      The size of the bird's vocab.
   * @param favoritePhrase The bird's favorite phrase
   */
  public RoseRingParakeet(String name, int vocabSize, String favoritePhrase) {
    super(name, false, 2, new ArrayList<>(Arrays.asList(Food.SEEDS, Food.VEGETATION)),
            vocabSize, favoritePhrase);
  }

}
