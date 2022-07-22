package disjointset;

/**
 * Implementation of a Disjoint set of nodes. Disjoint Set is an array of sets where nodes can only
 * exist in one of the sets.
 *
 * @param <T> A comparable type
 */
public interface DisjointSet<T> {
  /**
   * Finds the given node in the disjoint set.
   *
   * @param node The node to find
   * @return -1 if the node is not found otherwise the index in the array of the set it is in.
   */
  int findNode(T node);

  /**
   * Adds a node as a new set, does nothing if node is already in the disjoint set.
   *
   * @param node the node to add
   */
  void addSet(T node);

  /**
   * Merges the sets which contain the 2 nodes provided. Does nothing if the nodes are in the same
   * set already.
   *
   * @param node1 the first node to find and merge the sets
   * @param node2 the second node to merge and find the set of
   * @return true if the merge happens false if nothing happens
   */
  boolean mergeSet(T node1, T node2);
}
