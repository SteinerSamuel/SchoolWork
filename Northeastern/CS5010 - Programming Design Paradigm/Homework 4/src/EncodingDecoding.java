import java.util.Map;

/**
 * Implementation of encoding decoing using CodeTree.
 */
public interface EncodingDecoding {
  /**
   * Creates a hoffman prefix encoding dictionary given a message and a set of encoding strings.
   *
   * @param m         The message to generate the map from
   * @param encodeSet The set of tokens to use for encoding
   * @return a Map of the encodings in the order of symbol->code
   */
  Map<String, String> hoffmanEncoding(String m, String[] encodeSet);

  /**
   * Encodes a string given a dictionary.
   *
   * @param m         The string to encode
   * @param encodeSet A dictionary with  symbol->code  mapping.
   * @return
   */
  String encode(String m, Map<String, String> encodeSet);

  /**
   * Decodes a string given a dictionary.
   *
   * @param m         The String to decode
   * @param encodeSet A dictionary with the  symbol->code  mapping
   * @return The decoded string
   */
  String decode(String m, Map<String, String> encodeSet);
}
