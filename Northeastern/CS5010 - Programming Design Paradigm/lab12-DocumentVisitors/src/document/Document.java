package document;

import document.element.TextElementVisitor;

import java.util.ArrayList;
import java.util.List;

import document.element.TextElement;

/**
 * Document class.
 */
public class Document {

  private List<TextElement> content;

  /**
   * Default Contructor, makes an empty document.
   */
  public Document() {
    content = new ArrayList<>();
  }

  /**
   * Adds an element to the document.
   *
   * @param e The element to be added.
   */
  public void add(TextElement e) {
    content.add(e);
  }

  /**
   * Counts the words in the document.
   *
   * @return the word count
   */
  public int countWords() {
    int count = 0;
    for (TextElement c : content) {
      count = count + (Integer) c.accept(new WordCountVisitor());
    }
    return count;
  }

  /**
   * Text representation of the document based on the visitor passed.
   * @param visitor The visitor to make the text representation with.
   * @return the text representation of the document.
   */
  public String toText(TextElementVisitor visitor) {
    String delimiter = "\n";

    if (visitor instanceof BasicStringVisitor) {
      delimiter = " ";
    }

    StringBuilder sb = new StringBuilder();

    for (TextElement c : content) {
      sb.append(c.accept(visitor) + delimiter);
    }

    return sb.toString().trim();
  }
}
