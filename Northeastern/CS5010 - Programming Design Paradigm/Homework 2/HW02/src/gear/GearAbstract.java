package gear;

/**
 * An abstract class for gear to hold all functionality which will be found in all gear types.
 */
public abstract class GearAbstract implements Gear {
  protected Adjective prefix;
  protected Adjective combinedPrefix;
  protected int attack;
  protected int defense;
  protected boolean combine;

  @Override
  public int getAttack() {
    return attack;
  }

  @Override
  public int getDefense() {
    return defense;
  }

  @Override
  public boolean getCombined() {
    return combine;
  }

  /**
   * This is a method which will be used to help in the combination of the 2 pieces of gear.
   *
   * @return The prefix of the item.
   */
  protected Adjective getPrefix() {
    return this.prefix;
  }
}
