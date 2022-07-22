package character;

import gear.Gear;

/**
 * An interface for character this.
 */
public interface Character {
  /**
   * Get the name of the character.
   *
   * @return The name of the character,
   */
  String getName();

  /**
   * Gets the attack value of the character.
   *
   * @return The attack value of the character
   */
  int getAttack();

  /**
   * Gets the defense value of the character.
   *
   * @return The defense value of the character
   */
  int getDefense();

  /**
   * Equips a piece of gear.
   *
   * @param gear The gear which you are going to equip.
   */
  void equipGear(Gear gear);
}
