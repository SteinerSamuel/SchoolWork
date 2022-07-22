package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class which represents the owl classificaiton.
 */
public class Owl extends BirdAbstract {
  private final ArrayList<String> baseCharacteristics = new ArrayList<>(
          Arrays.asList("Soft Plumage", "Facial Disk"));

  /**
   * The default constructor for the owl class.
   *
   * @param name    The name of the bird.
   * @param extinct Is the bird extinct.
   * @param wings   the number of wings the bird has.
   * @param diet    The diet the bird has as a list of Food.
   */
  public Owl(String name, boolean extinct, int wings, ArrayList<Food> diet) {
    this.name = name;
    this.extinct = extinct;
    this.numOfWings = wings;
    this.dietPreference = diet;
    this.characteristics = baseCharacteristics;
  }

  /**
   * A constructor used if a owl has characteristics which are outside the base List.
   *
   * @param name                  The name of the bird.
   * @param extinct               Is the bird extinct.
   * @param wings                 the number of wings the bird has.
   * @param diet                  The diet the bird has as a list of Food.
   * @param extendCharacteristics A list of characteristics which extends the base.
   */
  public Owl(String name, boolean extinct, int wings, ArrayList<Food> diet,
             ArrayList<String> extendCharacteristics) {
    this.name = name;
    this.extinct = extinct;
    this.numOfWings = wings;
    this.dietPreference = diet;
    extendCharacteristics.addAll(baseCharacteristics);
    this.characteristics = extendCharacteristics;
  }

}
