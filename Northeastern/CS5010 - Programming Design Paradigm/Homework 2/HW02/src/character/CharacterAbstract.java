package character;

import gear.FeetGear;
import gear.Gear;
import gear.HandGear;
import gear.HeadGear;

/**
 * Class representation of a character.
 */
public class CharacterAbstract implements Character {
  protected final String name;
  protected final int attack;
  protected final int defense;
  protected HeadGear headslot = null;
  protected HandGear[] hands = new HandGear[2];
  protected FeetGear[] feets = new FeetGear[2];

  /**
   * Default constructor for character.
   *
   * @param name    the name of the character
   * @param attack  the base attack stat
   * @param defense the base defense stat
   */
  public CharacterAbstract(String name, int attack, int defense) {
    this.name = name;
    this.attack = attack;
    this.defense = defense;
  }

  @Override
  public String getName() {
    return name;
  }

  @Override
  public int getAttack() {
    int totalAttack = this.attack;

    for (HandGear hg : this.hands) {
      if (hg != null) {
        totalAttack += hg.getAttack();
      }
    }

    for (FeetGear fg : this.feets) {
      if (fg != null) {
        totalAttack += fg.getAttack();
      }
    }
    return totalAttack;
  }

  @Override
  public int getDefense() {
    int totalDefense = this.defense;

    if (this.headslot != null) {
      totalDefense += this.headslot.getDefense();
    }

    for (FeetGear fg : this.feets) {
      if (fg != null) {
        totalDefense += fg.getDefense();
      }
    }

    return totalDefense;
  }

  @Override
  public void equipGear(Gear gear) {
    int i = 0;
    boolean first_pass = true;
    if (gear instanceof HeadGear) {
      if (this.headslot == null) {
        this.headslot = (HeadGear) gear;
      } else if (!this.headslot.getCombined()) {
        this.headslot = (HeadGear) this.headslot.combine(gear);
      } else {
        throw new IllegalStateException("There is already an equipped helmet that has been combined"
        );
      }
    } else if (gear instanceof HandGear) {
      for (HandGear hg : hands) {
        if (hg == null) {
          this.hands[i] = (HandGear) gear;
          break;
        } else if (!hg.getCombined()) {
          this.hands[i] = (HandGear) hg.combine(gear);
          break;
        } else if (!first_pass) {
          throw new IllegalStateException("There is already 2 equipped pieces of gear which have "
                  + "been combined");
        }
        first_pass = false;
        i++;
      }
    } else if (gear instanceof FeetGear) {
      for (FeetGear hg : feets) {
        if (hg == null) {
          this.feets[i] = (FeetGear) gear;
          break;
        } else if (!hg.getCombined()) {
          this.feets[i] = (FeetGear) hg.combine(gear);
          break;
        } else if (!first_pass) {
          throw new IllegalStateException("There is already 2 equipped pieces of gear which have "
                  + "been combined");
        }
        first_pass = false;
        i++;
      }
    } else {
      throw new IllegalArgumentException("Unsupported gear type.");
    }
  }

  /**
   * To string prints a character sheet.
   *
   * @return the character sheet.
   */
  @Override
  public String toString() {
    return String.format("Character sheet for %s \n\tbase attack: %d\n\tbase defenese: %d"
                    + "\nEquipped Gear: \n\t%s\n\t%s\n\t%s\n\t%s\n\t%s"
                    + "\nFinal stats: \n\t%d\n\t%d", this.getName(), this.attack, this.defense,
            this.headslot.toString(), this.hands[0] != null ? this.hands[0].toString() : "Empty",
            this.hands[1] != null ? this.hands[1].toString() : "Empty",
            this.feets[0] != null ? this.feets[0].toString() : "Empty",
            this.feets[1] != null ? this.feets[1].toString() : "Empty", this.getAttack(),
            this.getDefense()
    );
  }
}
