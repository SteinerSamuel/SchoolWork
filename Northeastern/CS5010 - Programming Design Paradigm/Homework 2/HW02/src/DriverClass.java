import character.Character;
import character.CharacterAbstract;
import gear.Adjective;
import gear.FeetGear;
import gear.FeetNoun;
import gear.Gear;
import gear.HandGear;
import gear.HandNoun;
import gear.HeadGear;
import gear.HeadNoun;

/**
 * Driver class which showcases functionality of the project.
 */
public class DriverClass {

  public static final Character CHARACTER_1 = new CharacterAbstract("Dracula", 0, 0);

  public static final Character CHARACTER_2 = new CharacterAbstract("Richard", 0, 0);

  public static final Gear[] AVAILABLE_GEAR = {new HeadGear(Adjective.maddening, HeadNoun.Cap, 2),
      new HeadGear(Adjective.clever, HeadNoun.MidHelmet, 4),
      new HeadGear(Adjective.hallowed, HeadNoun.FullHelmet, 10),
      new HandGear(Adjective.sharp, HandNoun.BrassKnuckles, 1),
      new HandGear(Adjective.hallowed, HandNoun.Shield, 3),
      new HandGear(Adjective.enchanting, HandNoun.SpikedGloves, 8),
      new HandGear(Adjective.farflung, HandNoun.Spear, 8),
      new FeetGear(Adjective.common, FeetNoun.Boots, 2, 2),
      new FeetGear(Adjective.maddening, FeetNoun.CowboyBoots, 3, 1),
      new FeetGear(Adjective.maniacal, FeetNoun.Hoverboard, 1, 3)
  };

  /**
   * Equips a gear to a character.
   *
   * @param i the character to equip the gear to as 1 or 2
   * @param j the gear to equip as a number from 1 through 10 inclusive
   */
  public static void equip(int i, int j) {
    if (i < 1 || 2 < i) {
      throw new IllegalArgumentException("The character value must be 1 or 2");
    } else if (j < 1 || 10 < j) {
      throw new IllegalArgumentException("The equipment value must be between 1 through 10");
    } else if (i == 1) {
      CHARACTER_1.equipGear(AVAILABLE_GEAR[j - 1]);
    } else {
      CHARACTER_2.equipGear(AVAILABLE_GEAR[j - 1]);
    }
  }

  /**
   * The battle code this calculates the winner of the battle.
   */
  public static void battle() {
    int c1a = CHARACTER_1.getAttack();
    int c1d = CHARACTER_1.getDefense();
    int c2a = CHARACTER_2.getAttack();
    int c2d = CHARACTER_2.getDefense();
    int c1damage = c2a - c1d;
    int c2damage = c1a - c2d;

    if (c1damage < c2damage) {
      System.out.printf("%S is the winner!%n", CHARACTER_1.getName());
      System.out.printf("The final score was %s : %d damage, %s : %d damage", CHARACTER_1.getName(),
              c1damage, CHARACTER_2.getName(), c2damage);
    } else if (c2damage < c1damage) {
      System.out.printf("%S is the winner!%n", CHARACTER_2.getName());
      System.out.printf("The final score was %s : %d damage, %s : %d damage", CHARACTER_1.getName(),
              c1damage, CHARACTER_2.getName(), c2damage);
    } else {
      System.out.println("The battle ended in a draw!");
    }

  }

  /**
   * A rule for how gear is selected this gets the gear that is still available with the highest
   * attack stat.
   *
   * @param c The character as a number 1 or 2
   */
  private static void selectGear(int c) {
    int i = 0;
    int j = 0;
    int a = -999;
    for (Gear g : AVAILABLE_GEAR) {
      if (g != null && g.getAttack() > a) {
        a = g.getAttack();
        j = i;
      }
      i++;
    }
    System.out.printf("Character %x, will be selecting %s. %n", c, AVAILABLE_GEAR[j].getName());
    equip(c, j + 1);
    AVAILABLE_GEAR[j] = null;
  }

  /**
   * The main function which runs a showcase of the code.
   *
   * @param args none
   */
  public static void main(String[] args) {
    System.out.printf("%s and %s will be battling.%n", CHARACTER_1.getName(), CHARACTER_2.getName()
    );
    System.out.println("First they will select gear each selecting one piece till all gear is gone"
    );
    int i = 10;
    while (i > 0) {
      selectGear(i % 2 + 1);
      i--;
    }
    System.out.println("Here are the characters after picking");
    System.out.println(CHARACTER_1.toString());
    System.out.println(CHARACTER_2.toString());
    System.out.println("Now they battle.");

    battle();
  }

}
