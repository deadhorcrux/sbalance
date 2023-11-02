import org.pmml4s.model.Model;

import java.util.*;

public class Main {

    private final Model model = Model.fromFile(Main.class.getClassLoader().getResource("./models/v2/LGBMRV2Fold0.pmml").getFile());
}

