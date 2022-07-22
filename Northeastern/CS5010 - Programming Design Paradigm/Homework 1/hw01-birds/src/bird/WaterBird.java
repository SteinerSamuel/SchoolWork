package bird;

/**
 * A interface for birds which live near water.
 */
public interface WaterBird extends Bird {
  /**
   * Gets the body of water which the bird lives by.
   *
   * @return The body of water the bird lives by.
   */
  String getBodyOfWater();
}
