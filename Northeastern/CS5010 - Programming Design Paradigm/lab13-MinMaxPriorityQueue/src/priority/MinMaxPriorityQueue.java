package priority;

/**
 * A min max priority queue. A priority queue is a popular data structure. It is useful in many
 * applications (e.g. simulating customers at an airline counter) and algorithms (e.g. Dijkstra's
 * shortest path algorithm). A priority queue stores items, each of which is associated with a
 * priority. The priority is usually denoted as a single number. The main operations offered by a
 * priority queue are addition, and removal of the item of highest (or lowest) priority. A priority
 * queue can be implemented in several ways: one of the most popular is a heap data structure. The
 * binary heap allows addition and removal (the minimum or maximum priority, depending on
 * implementation) in O(log n), where n is the number of items in the queue.
 */
public interface MinMaxPriorityQueue<T> {
  /**
   * Adds an item to the MinMax priority queue based on a priority given.
   *
   * @param item     The item to be added
   * @param priority the priority of the value to be added
   */
  void add(T item, int priority);

  /**
   * Remove and return the item with the minimum priority (as defined by the lowest numeric priority
   * above). If such an item does not exist, this method should return null.
   *
   * @return the item with the minimum priority or null
   */
  T minPriorityItem();

  /**
   * Remove and return the item with the maximum priority (as defined by the highest numeric
   * priority above). If such an item does not exist, this method should return
   *
   * @return the tiem with the maximum priority or null
   */
  T maxPriorityItem();
}
