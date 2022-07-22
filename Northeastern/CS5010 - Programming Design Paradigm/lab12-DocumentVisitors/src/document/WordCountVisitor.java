package document;

import document.element.BasicText;
import document.element.BoldText;
import document.element.Heading;
import document.element.HyperText;
import document.element.ItalicText;
import document.element.Paragraph;
import document.element.TextElementVisitor;

/**
 * Visitor class for the word count of a document.
 */
public class WordCountVisitor implements TextElementVisitor {

  @Override
  public Object visitBasicText(BasicText current) {
    return current.getText().split("\\s+").length;
  }

  @Override
  public Object visitBoldText(BoldText current) {
    return current.getText().split("\\s+").length;
  }

  @Override
  public Object visitHeading(Heading current) {
    return current.getText().split("\\s").length;
  }

  @Override
  public Object visitHyperText(HyperText current) {
    return current.getText().split("\\s").length;
  }

  @Override
  public Object visitItalicText(ItalicText current) {
    return current.getText().split("\\s").length;
  }

  @Override
  public Object visitParagraph(Paragraph current) {
    return current.getText().split("\\s").length;
  }
}
