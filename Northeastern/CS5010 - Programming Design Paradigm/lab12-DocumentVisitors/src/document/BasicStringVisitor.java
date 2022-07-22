package document;

import document.element.BasicText;
import document.element.BoldText;
import document.element.Heading;
import document.element.HyperText;
import document.element.ItalicText;
import document.element.Paragraph;
import document.element.TextElementVisitor;

/**
 * Visitor for Document which is a basic string representation.
 */
public class BasicStringVisitor implements TextElementVisitor {
  @Override
  public Object visitBasicText(BasicText current) {
    return current.getText();
  }

  @Override
  public Object visitBoldText(BoldText current) {
    return current.getText();
  }

  @Override
  public Object visitHeading(Heading current) {
    return current.getText();
  }

  @Override
  public Object visitHyperText(HyperText current) {
    return current.getText();
  }

  @Override
  public Object visitItalicText(ItalicText current) {
    return current.getText();
  }

  @Override
  public Object visitParagraph(Paragraph current) {
    return current.getText();
  }
}
