package priority;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.TreeMap;

/**
 * Implementation of MinMaxPriority Queue.
 */
public class MinMaxPriorityQueueImpl<T> implements MinMaxPriorityQueue<T> {
  TreeMap<Integer, ArrayList<T>> sortedset = new TreeMap<Integer, ArrayList<T>>();

  @Override
  public void add(T item, int priority) {
    if (sortedset.get(priority) == null) {
      sortedset.put(priority, new ArrayList<T>(Arrays.asList(item)));
    } else {
      sortedset.get(priority).add(item);
    }
  }

  @Override
  public T minPriorityItem() {
    try {
      if (sortedset.firstEntry().getValue().size() == 1) {
        return sortedset.pollFirstEntry().getValue().remove(0);
      } else {
        return sortedset.firstEntry().getValue().remove(0);
      }
    } catch (NullPointerException npe) {
      return null;
    }
  }

  @Override
  public T maxPriorityItem() {
    try {
      if (sortedset.lastEntry().getValue().size() == 1) {
        return sortedset.pollLastEntry().getValue().remove(0);
      } else {
        return sortedset.lastEntry().getValue().remove(0);
      }
    } catch (NullPointerException npe) {
      return null;
    }
  }
}
