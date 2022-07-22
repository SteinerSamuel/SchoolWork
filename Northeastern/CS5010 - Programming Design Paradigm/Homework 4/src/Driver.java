import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;

/**
 * Main driver class.
 */
public class Driver {
  /**
   * Main class for driver.
   *
   * @param args arguments
   * @throws IOException If file does not exist
   */
  public static void main(String[] args) throws IOException {
    int argIndex = 0;
    boolean generateEncodeMapping = false;
    boolean encode = false;
    boolean decode = false;
    boolean outputFile = false;
    String m = null;
    String[] encodeTokens = new String[0];
    HashMap<String, String> encodeMap = new HashMap<>();

    while (argIndex < args.length) {
      switch (args[argIndex]) {
        case "-encode":
          encode = true;
          break;
        case "-decode":
          decode = true;
          break;
        case "-generate":
          generateEncodeMapping = true;
          break;
        case "-file":
          outputFile = true;
          break;
        case "-input":
          m = args[argIndex + 1];
          argIndex++;
          break;
        case "-inputF":
          m = Files.readString(Path.of(System.getProperty("user.dir")
                  + "/" + args[argIndex + 1]));
          argIndex++;
          break;
        case "-encodeTokens":
          encodeTokens = args[argIndex + 1].split(" ");
          argIndex++;
          break;
        case "-tokensF":
          encodeTokens = Files.readString(Path.of(System.getProperty("user.dir") + "/"
                  + args[argIndex + 1])).split("\n");
          argIndex ++;
          break;
        case "-encodeDict":
          String[] temp = args[argIndex + 1].split(" ");
          encodeMap = dictHelper(temp);
          argIndex++;
          break;
        case "-encodeDictF":
          String[] temp1 = Files.readString(Path.of(System.getProperty("user.dir") + "/"
                  + args[argIndex + 1])).split("\n");
          encodeMap = dictHelper(temp1);
          argIndex++;
          break;
        default:
          break;
      }
      argIndex++;
    }

    if ((encode ^ decode ^ generateEncodeMapping) ^ (encode && decode && generateEncodeMapping)) {
      EncodingDecoding ed = new EncodingDecodingImpl();
      if (m == null) {
        throw new IllegalArgumentException("Please include a message or file check docs for how!");
      }
      if ((decode || encode) && encodeMap.isEmpty()) {
        throw new IllegalArgumentException("Please provide an encoding map.");
      }
      if (generateEncodeMapping && encodeTokens.length < 1) {
        throw new IllegalArgumentException("Please provide a proper list of encoding tokens.");
      }
      String result = "";
      if (encode) {
        result = ed.encode(m, encodeMap);
      } else if (decode) {
        result = ed.decode(m, encodeMap);
      } else {
        result = ed.hoffmanEncoding(m, encodeTokens).toString();
      }

      if (outputFile) {
        File file = new File(System.getProperty("user.dir") + "/output.txt");
        file.createNewFile();
        FileWriter fw = new FileWriter(System.getProperty("user.dir") + "/output.txt");
        fw.write(result);
        fw.close();
      } else {
        System.out.println(result);
      }
    } else {
      throw new IllegalArgumentException("Select only 1 action!");
    }
  }

  private static HashMap<String, String> dictHelper(String[] temp) {
    HashMap<String, String> encodeMap = new HashMap<>();
    for (String code : temp) {
      if (code.split("=").length == 2) {
        encodeMap.put(code.split("=")[0].equals("") ? " " : code.split("=")[0],
                code.split("=")[1]);
      } else {
        encodeMap.put(" ", code.split("=")[0]);
      }
    }
    return encodeMap;
  }
}
