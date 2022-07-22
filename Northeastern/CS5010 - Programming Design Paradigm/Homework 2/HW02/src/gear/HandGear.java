package gear;

/**
 * Hand gear class, represents a piece of hand gear which only provides attack stats.
 */
public class HandGear extends GearAbstract {
  private final HandNoun suffix;

  /**
   * Default constructor, makes a Hand gear which makes an uncombined gear.
   *
   * @param prefix the adjective prefix
   * @param suffix the hand noun suffix
   * @param attack the attack provided by the item
   */
  public HandGear(Adjective prefix, HandNoun suffix, int attack) {
    if (prefix == null) {
      throw new IllegalArgumentException("Prefix must not be null!");
    } else if (suffix == null) {
      throw new IllegalArgumentException("Suffix must not be null!");
    }
    this.suffix = suffix;
    this.prefix = prefix;
    this.attack = attack;
    this.defense = 0;
    this.combinedPrefix = null;
    this.combine = false;
  }

  /**
   * The combined gear constructor  which makes a gear which has been combined.
   *
   * @param prefix         the adjective prefix
   * @param suffix         the hand noun suffix
   * @param attack         the attack provided by the item
   * @param combinedPrefix the combined prefix this will be the first word in the name
   */
  public HandGear(Adjective prefix, HandNoun suffix, int attack, Adjective combinedPrefix) {
    if (prefix == null) {
      throw new IllegalArgumentException("Prefix must not be null!");
    } else if (suffix == null) {
      throw new IllegalArgumentException("Suffix must not be null!");
    } else if (combinedPrefix == null) {
      throw new IllegalArgumentException("combinedPrefix must not be null!");
    }
    this.combine = true;
    this.combinedPrefix = combinedPrefix;
    this.prefix = prefix;
    this.attack = attack;
    this.defense = 0;
    this.suffix = suffix;
  }

  @Override
  public String getName() {
    if (combinedPrefix != null) {
      return String.format("%s, %s %s", combinedPrefix.toString(), prefix.toString(),
              suffix.toString());
    } else {
      return String.format("%s %s", prefix.toString(), suffix.toString());
    }
  }

  @Override
  public Gear combine(Gear gear2) {
    if (this.combine || gear2.getCombined()) {
      throw new IllegalArgumentException("You cannot combine these pieces of equipment, one of them"
              + "is already combined!");
    } else if (gear2 instanceof HandGear) {
      HandGear g2 = (HandGear) gear2;
      return new HandGear(this.prefix, this.suffix, this.attack + g2.getAttack(),
              g2.getPrefix());
    } else {
      throw new IllegalArgumentException("gear.Gear types much match!");
    }
  }

  @Override
  public String toString() {
    return String.format("%s provides %d attack bonus and no defense bonus", getName(), getAttack()
    );
  }
}
