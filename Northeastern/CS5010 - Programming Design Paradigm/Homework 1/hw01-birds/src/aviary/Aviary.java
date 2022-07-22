package aviary;

import bird.Bird;
import java.util.ArrayList;
import javax.naming.SizeLimitExceededException;



/**
 * Aviary interface.
 */
public interface Aviary {
  /**
   * Grabs a list of birds that are in the aviary.
   *
   * @return An ArrayList of birds.
   */
  ArrayList<Bird> getBirds();

  /**
   * Grabs the location of the aviary.
   *
   * @return A string which is the location of the aviary.
   */
  String getLocation();

  /**
   * Returns whether or not the aviary is empty.
   *
   * @return True if the aviary is empty, false if not empty.
   */
  boolean isEmpty();

  /**
   * Checks if a bird exists in an aviary.
   *
   * @param bird A bird to see if it exists in the aviary.
   * @return True if the bird is in the aviary, false if it is not in the aviary.
   */
  boolean hasBird(Bird bird);

  /**
   * Adds a bird to the aviary.
   *


   * @param bird a bird which should be added to the aviary.
   */
  void addBird(Bird bird) throws SizeLimitExceededException;
}
