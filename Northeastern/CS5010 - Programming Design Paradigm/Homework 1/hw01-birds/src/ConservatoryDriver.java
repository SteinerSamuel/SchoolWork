
import aviary.Aviary;
import bird.Bird;
import bird.Emu;
import bird.Hawk;
import bird.Kiwi;
import conservatory.Conservatory;
import conservatory.ConservatoryConcrete;
import java.util.ArrayList;

import javax.naming.SizeLimitExceededException;

public class ConservatoryDriver {
  /** The main driver class.
   *

   * @param args The args passed to the main method.
   */
  public static void main(String[] args) throws SizeLimitExceededException {
    Conservatory samsConservatory = new ConservatoryConcrete();
    ArrayList<Aviary> aviaries = new ArrayList<Aviary>(20);
    Bird herbert = new Hawk("Herbert");
    Bird hal = new Hawk("Hal");
    final Bird earl = new Emu("Earl");
    Bird kurt = new Kiwi("Kurt");

    aviaries.add(0, samsConservatory.addAviary("West Wing"));
    samsConservatory.addBird(herbert, aviaries.get(0));
    samsConservatory.addBird(hal, aviaries.get(0));

    aviaries.add(1, samsConservatory.addAviary("East Wing"));
    samsConservatory.addBird(earl, aviaries.get(1));

    //System.out.println(samsConservatory.calcFood());

    System.out.println(samsConservatory.getDirectory());

    System.out.println(samsConservatory.getIndex());
  }
}
