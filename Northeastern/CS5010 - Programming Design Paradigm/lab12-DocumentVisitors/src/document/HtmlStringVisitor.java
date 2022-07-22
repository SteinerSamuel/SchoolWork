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
 * HTML representation of a document.
 */
public class HtmlStringVisitor implements TextElementVisitor {
  @Override
  public Object visitBasicText(BasicText current) {
    return current.getText();
  }

  @Override
  public Object visitBoldText(BoldText current) {
    return "<b>" + current.getText() + "</b>";
  }

  @Override
  public Object visitHeading(Heading current) {
    return "<h" + current.getLevel() + ">" + current.getText() + "</h" + current.getLevel() + ">";
  }

  @Override
  public Object visitHyperText(HyperText current) {
    return "<a href=\"" + current.getUrl() + "\">" + current.getText() + "</a>";
  }

  @Override
  public Object visitItalicText(ItalicText current) {
    return "<i>" + current.getText() + "</i>";
  }

  @Override
  public Object visitParagraph(Paragraph current) {
    StringBuilder sb = new StringBuilder();

    for (TextElement c : current.getContent()) {
      sb.append(c.accept(this) + "\n");
    }

    return "<p>" + sb.toString() + "</p>";
  }
}
