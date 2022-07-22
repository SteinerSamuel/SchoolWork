package conservatory;

import aviary.Aviary;
import bird.Bird;
import java.util.ArrayList;

import javax.naming.SizeLimitExceededException;


/**
 * A conservatory implementation each conservatory can hold 20 aviaries.
 */
public interface Conservatory {
  /**
   * Grabs the list of aviaries.
   *
   * @return The list of aviaries.
   */
  ArrayList<Aviary> getAviaries();

  /**
   * Adds an aviary to the conservatory.
   *
   * @param location The location of the aviary.
   * @return The aviary that was created.
   */
  Aviary addAviary(String location) throws SizeLimitExceededException;

  /**
   * Adds a bird to the aviary.
   *
   * @param bird   The bird you want to add.
   * @param aviary The aviary you want to add the bird to.
   */
  void addBird(Bird bird, Aviary aviary) throws SizeLimitExceededException;

  /**
   * Calculates the food need.
   *
   * @return A string which shows the food needs of the conservatory.
   */
  String calcFood();

  /**
   * Finds the aviary which a bird belongs to.
   *
   * @param bird The bird which you want to find.
   * @return The Aviary which has the bird.
   */
  Aviary findBird(Bird bird);

  /**
   * Returns the description of the aviary, this has information on each of the birds in the
   * Aviary.
   *
   * @param aviary The aviary you want the description of.
   * @return A string of the description of the aviary.
   */
  String aviaryDescription(Aviary aviary);

  /**
   * Returns the Directory which is all aviaries in the conservatory.
   *
   * @return A string which is the directory of the aviary.
   */
  String getDirectory();

  /**
   * Lists all birds in the conservatory in alphabetical order and their location.
   *
   * @return The string which provides a list of all the birds in alphabetical order.
   */
  String getIndex();
}
