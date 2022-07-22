package document;

import document.element.BasicText;
import document.element.BoldText;
import document.element.Heading;
import document.element.HyperText;
import document.element.ItalicText;
import document.element.Paragraph;
import document.element.TextElement;
import document.element.TextElementVisitor;

/**
 * Visitor class for a document that is a markdown representation of the document.
 */
public class MarkdownStringVisitor implements TextElementVisitor {
  @Override
  public Object visitBasicText(BasicText current) {
    return current.getText();
  }

  @Override
  public Object visitBoldText(BoldText current) {
    return "**" + current.getText() + "**";
  }

  @Override
  public Object visitHeading(Heading current) {
    return "#".repeat(current.getLevel()) + " " + current.getText();
  }

  @Override
  public Object visitHyperText(HyperText current) {
    return "[" + current.getText() + "](" + current.getUrl() + ")";
  }

  @Override
  public Object visitItalicText(ItalicText current) {
    return "*" + current.getText() + "*";
  }

  @Override
  public Object visitParagraph(Paragraph current) {
    StringBuilder sb = new StringBuilder();

    for (TextElement c : current.getContent()) {
      sb.append(c.accept(this) + "\n");
    }

    return "\n" + sb.toString().trim();
  }
}
