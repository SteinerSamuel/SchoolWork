package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class which represents the birds of pray.
 */
public class BirdOfPray extends BirdAbstract {
  private final ArrayList<String> baseCharacteristics = new ArrayList<>(
          Arrays.asList("Large", "Sharp beak", "Visible nostrils"));

  /**
   * The default constructor for the Birds of Pray class.
   *
   * @param name    The name of the bird.
   * @param extinct Is the bird extinct.
   * @param wings   the number of wings the bird has.
   * @param diet    The diet the bird has as a list of Food.
   */
  public BirdOfPray(String name, boolean extinct, int wings, ArrayList<Food> diet) {
    this.name = name;
    this.extinct = extinct;
    this.numOfWings = wings;
    this.dietPreference = diet;
    this.characteristics = baseCharacteristics;
  }

  /**
   * A constructor used if a bird of pray has characteristics which are outside the base list.
   *
   * @param name                  The name of the bird
   * @param extinct               Is the bird extinct.
   * @param wings                 the number of wings the bird has.
   * @param diet                  The diet the bird has as a list of Food.
   * @param extendCharacteristics A list of characteristics which extend the base characteristics.
   */
  public BirdOfPray(String name, boolean extinct, int wings, ArrayList<Food> diet,
                    ArrayList<String> extendCharacteristics) {
    this.name = name;
    this.extinct = extinct;
    this.numOfWings = wings;
    this.dietPreference = diet;
    extendCharacteristics.addAll(baseCharacteristics);
    this.characteristics = extendCharacteristics;
  }

}
