package gear;

/**
 * Interface for gear which character will wear provides stat bonuses based on what gear slot the
 * gear is equipped to.
 */
public interface Gear {
  /**
   * Gets the name of the piece of gear.
   *
   * @return The name of the piece of gear as a string.
   */
  String getName();

  /**
   * Gets the attack bonus provided by the piece of gear.
   *
   * @return The attack bonus.
   */
  int getAttack();

  /**
   * Gets the defensive bonus provided by the piece of gear.
   *
   * @return The defensive bonus.
   */
  int getDefense();

  /**
   * Gets whether or not the piece of gear has been combined with another or not.
   *
   * @return True if the gear has been combined False if the gear has not been combined
   */
  boolean getCombined();

  /**
   * Combines a second piece of gear with the current one, can only happen if the gear hasn't been
   * combined yet.
   *
   * @param gear2 The gear to combine to the current gear.
   * @return The new combined gear.
   */
  Gear combine(Gear gear2);

}
