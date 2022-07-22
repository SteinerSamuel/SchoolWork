import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.TreeMap;

/**
 * Implementation of EncodingDecoding.
 */
public class EncodingDecodingImpl implements EncodingDecoding {

  @Override
  public Map<String, String> hoffmanEncoding(String message, String[] encodingSet) {
    // Generate frequency table
    HashMap<String, CodeTreeNode> frequencyTable = new HashMap<>();

    for (Character c : message.toCharArray()) {
      if (frequencyTable.containsKey(c.toString())) {
        frequencyTable.get(c.toString()).increaseFreq();
      } else {
        frequencyTable.put(c.toString(), new CodeTreeNode(c.toString()));
      }
    }
    PriorityQueue<Map.Entry<String, CodeTreeNode>> pq = new PriorityQueue<>(
        (a, b) -> a.getValue().getFreq() == b.getValue().getFreq()
                ? a.getKey().compareTo(b.getKey())
                : a.getValue().getFreq() - b.getValue().getFreq()
    );

    for (Map.Entry<String, CodeTreeNode> entry : frequencyTable.entrySet()) {
      pq.offer(entry);
    }

    boolean flag = true;
    while (flag) {
      ArrayList<CodeTreeNode> children = new ArrayList<CodeTreeNode>();
      StringBuilder token = new StringBuilder();
      int freq = 0;
      for (String encoder : encodingSet) {
        Map.Entry<String, CodeTreeNode> x = pq.poll();
        if (x == null) {
          break;
        }
        x.getValue().setRepresentation(encoder);
        children.add(x.getValue());
        token.append(x.getValue().getToken());
        freq += x.getValue().getFreq();
      }

      pq.offer(new AbstractMap.SimpleEntry<>(token.toString(), new CodeTreeNode(token.toString(),
              freq, children)));

      flag = pq.size() != 1;
    }

    Map.Entry<String, CodeTreeNode> root = pq.poll();

    return root.getValue().returnMapDict();
  }

  @Override
  public String encode(String m, Map<String, String> encodeSet) {
    StringBuilder encodedString = new StringBuilder();
    for (Character c : m.toCharArray()) {
      if (encodeSet.get(c.toString()) == null) {
        throw new IllegalArgumentException("Encoding set does not include all the characters which "
                + "appear in the message!");
      }
      encodedString.append(encodeSet.get(c.toString()));
    }
    return encodedString.toString();
  }

  @Override
  public String decode(String m, Map<String, String> encodeSet) {
    TreeMap<String, String> tm = new TreeMap<>();
    for (Map.Entry<String, String> entry : encodeSet.entrySet()) {
      tm.put(entry.getValue(), entry.getKey());
    }

    StringBuilder decodedSting = new StringBuilder();

    while (!m.isEmpty()) {
      int i = 1;
      String prefix = m.substring(0, i);
      while (tm.subMap(prefix, prefix + Character.MAX_VALUE).size() != 1) {
        i++;
        prefix = m.substring(0, i);
      }
      decodedSting.append(tm.get(prefix));
      m = m.substring(i);
    }
    return decodedSting.toString();
  }
}

