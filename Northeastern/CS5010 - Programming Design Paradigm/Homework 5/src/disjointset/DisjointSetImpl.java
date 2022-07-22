package disjointset;

import java.util.ArrayList;
import java.util.Collections;
import java.util.TreeSet;

/**
 * An implementation of a disjoint set which uses Tree Sets.
 *

 * @param <T> A comparable type
 */
public class DisjointSetImpl<T> implements DisjointSet<T> {
  ArrayList<TreeSet<T>> forest = new ArrayList<TreeSet<T>>();

  @Override
  public int findNode(T node) {
    int i = 0;

    for (TreeSet<T> t : forest) {
      if (t.contains(node)) {
        return i;
      }
      i++;
    }

    return -1;
  }

  @Override
  public void addSet(T node) {
    if (findNode(node) == -1) {
      forest.add((TreeSet<T>) new TreeSet<T>(Collections.singleton(node)));
    }
  }

  @Override
  public boolean mergeSet(T node, T node2) {
    int v = findNode(node);
    int w = findNode(node2);
    if (v != w && v >= 0 && w >= 0) {
      forest.get(v).addAll(forest.get(w));
      forest.remove(w);
      return true;
    }
    return false;
  }

  @Override
  public String toString() {
    StringBuilder s = new StringBuilder();
    for (TreeSet t : forest) {
      s.append(" ").append(t.toString());
    }
    return s.toString();
  }
}
