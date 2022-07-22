package gear;

/**
 * The feetGear class which represents the gear which could be equipped to the feet slots.
 */
public class FeetGear extends GearAbstract {
  private final FeetNoun suffix;

  /**
   * Default constrcutor this is used to create a basic feet gear item which is not combined.
   *
   * @param prefix  The adjective prefix
   * @param suffix  The noun suffix selected from the feetnoun list
   * @param defense the defense provided by the item
   * @param attack  the attack provided by the item
   */
  public FeetGear(Adjective prefix, FeetNoun suffix, int defense, int attack) {
    if (prefix == null) {
      throw new IllegalArgumentException("Prefix must not be null!");
    } else if (suffix == null) {
      throw new IllegalArgumentException("Suffix must not be null!");
    }
    this.suffix = suffix;
    this.prefix = prefix;
    this.defense = defense;
    this.attack = attack;
    this.combinedPrefix = null;
    this.combine = false;
  }

  /**
   * The combined constructor is used for when you want to create a combined item.
   *
   * @param prefix         The adjective prefix this goes as the second word of the name of the
   *                       item
   * @param suffix         the suffix feet noun of the item
   * @param defense        the defense provided by the item
   * @param attack         the attack provided by the item
   * @param combinedPrefix the combined prefix of the item this is used as the first word in the
   *                       name
   */
  public FeetGear(Adjective prefix, FeetNoun suffix, int defense, int attack,
                  Adjective combinedPrefix) {
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
    this.defense = defense;
    this.attack = attack;
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
    } else if (gear2 instanceof FeetGear) {
      FeetGear g2 = (FeetGear) gear2;
      return new FeetGear(this.prefix, this.suffix, this.defense + g2.getDefense(),
              this.attack + g2.getAttack(), g2.getPrefix());
    } else {
      throw new IllegalArgumentException("gear.Gear types much match!");
    }
  }

  @Override
  public String toString() {
    return String.format("%s provides %d attack bonus and %d defense bonus", getName(), getAttack(),
            getDefense());
  }
}
