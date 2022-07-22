package gear;

/**
 * The head geat class used to represent a piece of gear which goes in the head slot.
 */
public class HeadGear extends GearAbstract {
  private final HeadNoun suffix;

  /**
   * The default constructor creates a non combined gear for the head slot. Head gear does not
   * provide any attack bonus.
   *
   * @param prefix  the adjective prefix
   * @param suffix  the head noun suffix
   * @param defense the defense provided
   */
  public HeadGear(Adjective prefix, HeadNoun suffix, int defense) {
    if (prefix == null) {
      throw new IllegalArgumentException("Prefix must not be null!");
    } else if (suffix == null) {
      throw new IllegalArgumentException("Suffix must not be null!");
    }
    this.suffix = suffix;
    this.prefix = prefix;
    this.defense = defense;
    this.attack = 0;
    this.combinedPrefix = null;
    this.combine = false;
  }

  /**
   * The combined item constructor used to make an item which is combined.
   *
   * @param prefix         the adjective prefix
   * @param suffix         the head noun suffix
   * @param defense        the defense provided by the item
   * @param combinedPrefix the combined prefix this is the first word of the items name
   */
  public HeadGear(Adjective prefix, HeadNoun suffix, int defense, Adjective combinedPrefix) {
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
    this.attack = 0;
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
    } else if (gear2 instanceof HeadGear) {
      HeadGear g2 = (HeadGear) gear2;
      return new HeadGear(this.prefix, this.suffix, this.defense + g2.getDefense(),
              g2.getPrefix());
    } else {
      throw new IllegalArgumentException("gear.Gear types much match!");
    }
  }

  @Override
  public String toString() {
    return String.format("%s provides no attack bonus and %d defense bonus", getName(),
            getDefense());
  }
}
