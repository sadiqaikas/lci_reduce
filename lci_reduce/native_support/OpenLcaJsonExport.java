import java.io.File;
import java.util.HashSet;
import java.util.Set;

import org.openlca.core.database.Derby;
import org.openlca.core.database.ParameterDao;
import org.openlca.core.database.upgrades.Upgrades;
import org.openlca.core.database.upgrades.VersionState;
import org.openlca.core.model.ModelType;
import org.openlca.core.model.Parameter;
import org.openlca.core.model.RootEntity;
import org.openlca.core.model.descriptors.Descriptor;
import org.openlca.jsonld.ZipStore;
import org.openlca.jsonld.output.JsonExport;

public class OpenLcaJsonExport {

  private static int exportedCount = 0;

  public static void main(String[] args) throws Exception {
    if (args.length < 2 || args.length > 3) {
      System.err.println("usage: OpenLcaJsonExport <native-db-dir> <output-jsonld-zip> [full|inspect]");
      System.exit(2);
    }

    File dbDir = new File(args[0]);
    File outFile = new File(args[1]);
    String scope = args.length >= 3 ? args[2] : "full";
    if (!dbDir.isDirectory()) {
      throw new IllegalArgumentException("Database directory does not exist: " + dbDir);
    }
    var parent = outFile.getParentFile();
    if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
      throw new IllegalStateException("Failed to create output directory: " + parent);
    }
    if (outFile.exists() && !outFile.delete()) {
      throw new IllegalStateException("Failed to overwrite output file: " + outFile);
    }

    try (Derby db = new Derby(dbDir); ZipStore store = ZipStore.open(outFile)) {
      VersionState state = VersionState.get(db);
      if (state == VersionState.NEEDS_UPGRADE) {
        Upgrades.on(db);
      } else if (state == VersionState.HIGHER_VERSION) {
        throw new IllegalStateException("Database schema is newer than the installed openLCA runtime");
      } else if (state == VersionState.ERROR) {
        throw new IllegalStateException("Failed to determine native database schema version");
      }

      JsonExport export = new JsonExport(db, store).withDefaultProviders(true);
      Set<String> seen = new HashSet<>();

      for (ModelType type : ModelType.values()) {
        if (type == ModelType.PARAMETER) {
          continue;
        }
        if (!shouldExport(type, scope)) {
          continue;
        }
        Class<?> modelClass = type.getModelClass();
        if (modelClass == null || !RootEntity.class.isAssignableFrom(modelClass)) {
          continue;
        }
        exportType(db, export, modelClass, seen);
      }

      ParameterDao parameterDao = new ParameterDao(db);
      for (Parameter parameter : parameterDao.getGlobalParameters()) {
        if (parameter == null || parameter.refId == null || parameter.refId.isBlank()) {
          continue;
        }
        if (seen.add("PARAMETER:" + parameter.refId)) {
          export.write(parameter);
          onExport(parameter.refId);
        }
      }
    }
  }

  @SuppressWarnings({ "rawtypes", "unchecked" })
  private static void exportType(Derby db, JsonExport export, Class modelClass, Set<String> seen) {
    for (Object descriptorObject : db.getDescriptors(modelClass)) {
      if (!(descriptorObject instanceof Descriptor descriptor)) {
        continue;
      }
      if (descriptor.refId == null || descriptor.refId.isBlank()) {
        continue;
      }
      String key = modelClass.getSimpleName() + ":" + descriptor.refId;
      if (!seen.add(key)) {
        continue;
      }
      Object entity = db.get(modelClass, descriptor.refId);
      if (entity instanceof RootEntity root) {
        export.write(root);
        onExport(descriptor.refId);
      }
    }
  }

  private static void onExport(String refId) {
    exportedCount++;
    if (exportedCount % 500 == 0) {
      System.err.println("exported " + exportedCount + " datasets; last=" + refId);
      System.err.flush();
    }
  }

  private static boolean shouldExport(ModelType type, String scope) {
    if ("full".equalsIgnoreCase(scope)) {
      return true;
    }
    if (!"inspect".equalsIgnoreCase(scope)) {
      throw new IllegalArgumentException("Unsupported export scope: " + scope);
    }
    return switch (type) {
      case PROCESS, FLOW, IMPACT_METHOD, IMPACT_CATEGORY, FLOW_PROPERTY, UNIT_GROUP -> true;
      default -> false;
    };
  }
}
