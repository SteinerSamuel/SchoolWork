package conservatory;

import aviary.Aviary;
import aviary.AviaryConcrete;
import bird.Bird;
import bird.BirdOfPray;
import bird.FlightlessBird;
import bird.Food;
import bird.Waterfowl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.StringJoiner;

import javax.naming.SizeLimitExceededException;


public final class ConservatoryConcrete implements Conservatory {
  private ArrayList<Aviary> aviaries = new ArrayList<Aviary>(20);

  @Override
  public ArrayList<Aviary> getAviaries() {
    return aviaries;
  }

  @Override
  public Aviary addAviary(String location) throws SizeLimitExceededException {
    if (aviaries.size() < 20) {
      Aviary a = new AviaryConcrete(location);
      aviaries.add(a);
      return a;
    } else {
      throw new SizeLimitExceededException("Conservatory can only have 20 aviaries.");
    }
  }

  @Override
  public void addBird(Bird bird, Aviary aviary) throws SizeLimitExceededException {
    if (this.aviaries.contains(aviary)) {
      // Checks to see if the bird is extinct.
      if (bird.getExtinct()) {
        throw new IllegalArgumentException("The bird you are trying to add is extinct.");
      } else {
        // If the aviary is empty you can just add the bid
        if (aviary.isEmpty()) {
          aviary.addBird(bird);
          // Check to see if any of the birds must stay within its classification if not just add
          // the bird
        } else if (!((aviary.getBirds().get(0) instanceof BirdOfPray
                || aviary.getBirds().get(0) instanceof FlightlessBird
                || aviary.getBirds().get(0) instanceof Waterfowl)
                && (bird instanceof BirdOfPray
                || bird instanceof FlightlessBird
                || bird instanceof Waterfowl))) {
          aviary.addBird(bird);
          // If the birds match then we can just add the bird
        } else if ((aviary.getBirds().get(0) instanceof BirdOfPray && bird instanceof BirdOfPray)
                || (aviary.getBirds().get(0) instanceof FlightlessBird
                && bird instanceof FlightlessBird)
                || (aviary.getBirds().get(0) instanceof Waterfowl && bird instanceof Waterfowl)) {
          aviary.addBird(bird);
          // If we reach here the birds are incompatible.
        } else {
          throw new IllegalArgumentException("This bird cannot be added because of a restriction on"
                  + " the aviary please check the aviary.");
        }

      }
    }
  }


  @Override
  public String calcFood() {
    HashMap<Food, Integer> dietMap = new HashMap<Food, Integer>();
    for (Aviary a : aviaries) {
      for (Bird b : a.getBirds()) {
        for (Food f : b.getDietPreference()) {
          if (dietMap.containsKey(f)) {
            dietMap.put(f, dietMap.get(f) + 1);
          } else {
            dietMap.put(f, 1);
          }
        }
      }
    }
    return dietMap.toString();
  }

  @Override
  public Aviary findBird(Bird bird) {
    for (Aviary a : aviaries) {
      if (a.hasBird(bird)) {
        return a;
      }
    }
    return null;
  }

  @Override
  public String aviaryDescription(Aviary aviary) {
    if (aviaries.contains(aviary)) {
      return aviary.toString();
    } else {
      throw new IllegalArgumentException("That aviary doesn't exist in this conservatory please "
              + "check again, or add it.");
    }
  }

  @Override
  public String getDirectory() {
    StringBuilder directory = new StringBuilder();
    for (Aviary a : aviaries) {
      directory.append(a.toString()).append('\n');
    }
    return directory.toString();
  }

  @Override
  public String getIndex() {
    // add all the birds to a list
    ArrayList<String> names = new ArrayList<>(100);
    for (Aviary a : aviaries) {
      for (Bird b : a.getBirds()) {
        names.add(b.getBirdName() + '.' + a.getLocation());
      }
    }
    // sort the list
    Collections.sort(names);

    // format the list into a neat stirng
    StringJoiner indexJoiner = new StringJoiner(", \n");

    for (String s : names) {
      indexJoiner.add(s.split("\\.")[0] + " is in the aviary located at "
              + s.split("\\.")[1]);
    }
    return indexJoiner.toString();
  }
}
