package aviary;

import bird.Bird;
import java.util.ArrayList;
import java.util.StringJoiner;
import javax.naming.SizeLimitExceededException;


public final class AviaryConcrete implements Aviary {
  private final ArrayList<Bird> listOfBirds = new ArrayList<>(5);
  private final String location;

  public AviaryConcrete(String location) {
    this.location = location;
  }

  @Override
  public ArrayList<Bird> getBirds() {
    return this.listOfBirds;
  }

  @Override
  public String getLocation() {
    return this.location;
  }

  @Override
  public boolean isEmpty() {
    return this.listOfBirds.isEmpty();
  }

  @Override
  public boolean hasBird(Bird bird) {
    return this.listOfBirds.contains(bird);
  }

  @Override
  public void addBird(Bird bird) throws SizeLimitExceededException {
    if (listOfBirds.size() < 5) {
      this.listOfBirds.add(bird);
    } else {
      throw new SizeLimitExceededException("The size of an aviary is 5, you cannot add more"
              + "birds to the aviary.");
    }
  }

  @Override
  public String toString() {
    StringJoiner birds = new StringJoiner("");
    for (Bird b : listOfBirds) {
      birds.add(b.toString());
    }
    return "This is the aviary located at " + location
            + ". The following birds are in this aviary:" + "\n" + birds.toString();
  }
}
