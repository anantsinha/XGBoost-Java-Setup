import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoost;

public class Main {
    public static void main(String[] args) throws XGBoostError {
        Booster booster = XGBoost.loadModel("xgbModel.model");
        DMatrix dtest = new DMatrix("libsvmData.txt");
        System.out.println("Loaded data");
        float[][] predicts = booster.predict(dtest);
        // nth predicted value is extracted by predicts[n][0]
        System.out.println("First predicted value is:");
        System.out.println(predicts[0][0]);
    }
}