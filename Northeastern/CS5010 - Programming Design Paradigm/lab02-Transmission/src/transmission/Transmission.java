package transmission;

/**
 * Transmission interface, this is an interface which represents a transmission state. Each instance
 * of classes which this interface is implemented only represents a single state of the
 * transmission.
 */
public interface Transmission {
  /**
   * Returns the speed of the current Transmission state.
   *
   * @return The speed of the transmission.
   */
  int getSpeed();

  /**
   * Returns the current gear of the Transmission.
   *
   * @return The gear of th transmission.
   */
  int getGear();

  /**
   * Adds 2 to the speed and returns a transmission in the correct gear.
   *
   * @return The transmission in the new state.
   */
  Transmission increaseSpeed();

  /**
   * Decreases the speed by 2 and returns the transmission in the correct gear.
   *
   * @return The Transmission in the new state.
   */
  Transmission decreaseSpeed();
}
