import java.util.HashMap;
import java.util.List;

/**
 * Implementation of Code Tree.
 */
public class CodeTreeNode implements CodeTree {
  Integer frequency;
  private List<CodeTreeNode> children;
  private String token;
  private String representation;

  /**
   * Makes a node with just a token.
   *
   * @param token the token which the node holds
   */
  public CodeTreeNode(String token) {
    this.token = token;
    this.frequency = 1;
    this.representation = "";
  }

  /**
   * A constructor for children and frequency.
   *
   * @param token     The token the node holds
   * @param frequency the frequency of the token
   * @param children  The children of the node
   */
  public CodeTreeNode(String token, Integer frequency, List<CodeTreeNode> children) {
    this.token = token;
    this.frequency = frequency;
    this.children = children;
    this.representation = "";
  }

  @Override
  public void increaseFreq() {
    this.frequency++;
  }

  @Override
  public int getFreq() {
    return frequency;
  }


  @Override
  public String getToken() {
    return token;
  }

  @Override
  public String getRepresentation() {
    return representation;
  }

  @Override
  public void setRepresentation(String rep) {
    this.representation = rep;
  }

  @Override
  public HashMap<String, String> returnMapDict() {
    HashMap<String, String> mapDict = new HashMap<>();
    if (children != null) {
      for (CodeTreeNode child : children) {
        child.setRepresentation(representation + child.getRepresentation());
        mapDict.putAll(child.returnMapDict());
      }
    } else {
      mapDict.put(token, representation);
    }
    return mapDict;
  }
}
