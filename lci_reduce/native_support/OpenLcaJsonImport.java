import java.io.File;

import org.openlca.core.database.Derby;
import org.openlca.jsonld.ZipStore;
import org.openlca.jsonld.input.JsonImport;

public class OpenLcaJsonImport {

  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      System.err.println("usage: OpenLcaJsonImport <jsonld-zip> <native-db-dir>");
      System.exit(2);
    }

    File jsonldZip = new File(args[0]);
    File dbDir = new File(args[1]);
    if (!jsonldZip.isFile()) {
      throw new IllegalArgumentException("JSON-LD ZIP does not exist: " + jsonldZip);
    }
    if (dbDir.exists()) {
      throw new IllegalArgumentException("Native database directory already exists: " + dbDir);
    }
    if (!dbDir.mkdirs()) {
      throw new IllegalStateException("Failed to create native database directory: " + dbDir);
    }

    try (Derby db = new Derby(dbDir); ZipStore store = ZipStore.open(jsonldZip)) {
      JsonImport importer = new JsonImport(store, db);
      importer.run();
    }
  }
}
