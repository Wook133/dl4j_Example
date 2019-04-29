package Iris;

public class DeepLearningApp {
    public static void main(String[] args) throws Exception {
        IrisClassifier classifier = new IrisClassifier();
        classifier.classify("iris.csv", "iris-test.csv");
    }
}
