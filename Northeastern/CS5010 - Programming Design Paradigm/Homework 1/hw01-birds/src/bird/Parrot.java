package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class which represents the parrot classificaiton.
 */
public class Parrot extends TalkingBirdAbstract {
  private final ArrayList<String> baseCharacteristics = new ArrayList<>(
          Arrays.asList("Short beak", "Curved beak", "Intelligent", "Mimics sounds"));

  /**
   * The default constructor for the parrot class.
   *
   * @param name           The name of the bird.
   * @param extinct        Is the bird extinct.
   * @param wings          the number of wings the bird has.
   * @param diet           The diet the bird has as a list of Food.
   * @param vocabSize      The size of the birds vocab.
   * @param favoritePhrase The bird's favorite phrase.
   */
  public Parrot(String name, boolean extinct, int wings, ArrayList<Food> diet, int vocabSize,
                String favoritePhrase) {
    this.name = name;
    this.extinct = extinct;
    this.numOfWings = wings;
    this.dietPreference = diet;
    this.characteristics = baseCharacteristics;
    this.vocabSize = vocabSize;
    this.favoritePhrase = favoritePhrase;

  }

  /**
   * A constructor used if a parrot has characteristics which are outside the base List.
   *
   * @param name                  The name of the bird.
   * @param extinct               Is the bird extinct.
   * @param wings                 the number of wings the bird has.
   * @param diet                  The diet the bird has as a list of Food.
   * @param vocabSize             The Size of the birds vocab.
   * @param favoritePhrase        The bird's favorite phrase.
   * @param extendCharacteristics A list of characteristics which extends the base.
   */
  public Parrot(String name, boolean extinct, int wings, ArrayList<Food> diet,
                int vocabSize, String favoritePhrase, ArrayList<String> extendCharacteristics) {
    this.extinct = extinct;
    this.numOfWings = wings;
    this.dietPreference = diet;
    extendCharacteristics.addAll(baseCharacteristics);
    this.characteristics = extendCharacteristics;
    this.vocabSize = vocabSize;
    this.favoritePhrase = favoritePhrase;
  }

}
