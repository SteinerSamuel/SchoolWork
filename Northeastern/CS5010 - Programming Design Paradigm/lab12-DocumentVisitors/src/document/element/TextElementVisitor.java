package document.element;

/**
 * Text element visitor interface.
 *
 * @param <R> the type of object that the visitor returns
 */
public interface TextElementVisitor<R> {

  R visitBasicText(BasicText current);

  R visitBoldText(BoldText current);

  R visitHeading(Heading current);

  R visitHyperText(HyperText current);

  R visitItalicText(ItalicText current);

  R visitParagraph(Paragraph current);
}
