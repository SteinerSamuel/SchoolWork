import org.junit.Test;

import java.util.HashMap;
import java.util.Map;
import static org.junit.Assert.assertEquals;

/**
 * Test for encode decode.
 */
public class EncodeDecodeTest {
  @Test
  public void test() {
    String m = "she sells sea shells by the sea shore";
    EncodingDecodingImpl ed = new EncodingDecodingImpl();
    Map<String, String> hm = ed.hoffmanEncoding(m,
            new String[]{"0", "1"});

    HashMap<String, String> test = new HashMap<>();
    test.put(" ","110");
    test.put("a","0001");
    test.put("b","00100");
    test.put("r","00110");
    test.put("s","10");
    test.put("t","00111");
    test.put("e","111");
    test.put("h","010");
    test.put("y","0000");
    test.put("l","011");
    test.put("o","00101");

    assertEquals(test, hm);

    String encoded = ed.encode("she sells sea shells by the sea shore", hm);

    assertEquals("10010111110101110110111011010111000111010010111011011101100010000001100"
            + "0111010111110101110001110100100010100110111", encoded);

    String decoded = ed.decode(encoded, hm);

    assertEquals(m, decoded);
  }
}