package Regression;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;

public class eg3 {
    public static final int nEpochs = 100;
    public static final int plotFrequency = 5;
    public static void main(String[] args) {
        int numfeatures = 5;
        int numRegressors = 1;
        int datasetSize = 1460;
        int seed = 42;
        double splitrate = 0.7;


        try
        {
            //Vectorizing and Reading the Data
            int numLinesSkip = 0;
            String delimin = ",";
            RecordReader recordReaderTrain = new CSVRecordReader(numLinesSkip, delimin);
            recordReaderTrain.initialize(new FileSplit(
                    new ClassPathResource("trainx.txt").getFile()));
            DataSetIterator iteratorTrain = new RecordReaderDataSetIterator(
                    recordReaderTrain, datasetSize, numfeatures, numRegressors);
            System.out.println(iteratorTrain.inputColumns());
            System.out.println(iteratorTrain.totalOutcomes());

            RecordReader recordReaderTest = new CSVRecordReader(numLinesSkip, delimin);
            recordReaderTest.initialize(new FileSplit(
                    new ClassPathResource("testx.txt").getFile()));
            DataSetIterator iteratorTest = new RecordReaderDataSetIterator(
                    recordReaderTest, 230, numfeatures, numRegressors);

            RecordReader recordReaderValidation = new CSVRecordReader(numLinesSkip, delimin);
            recordReaderTest.initialize(new FileSplit(
                    new ClassPathResource("validx.txt").getFile()));
            DataSetIterator iteratorValidation = new RecordReaderDataSetIterator(
                    recordReaderValidation, 230, numfeatures, numRegressors);


            //Build Network
           /* MultiLayerConfiguration configuration
                    = new NeuralNetConfiguration.Builder()
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    //.l1(0.1)

                    .l2(0.0001)
                    .updater(Updater.SGD)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(numfeatures).nOut(3).build())
                    .layer(1, new DenseLayer.Builder().nIn(3).nOut(10).build())
                    .layer(2, new DenseLayer.Builder().nIn(10).nOut(3).build())
                    .layer(3, new OutputLayer.Builder(
                            LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX)
                            .nIn(3).nOut(numRegressors).build())
                    .backprop(true).pretrain(false)
                    .build();*/
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .weightInit(WeightInit.XAVIER)
                    .updater(Updater.ADAGRAD)
                    .activation(Activation.RELU)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .l2(0.0001)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(numfeatures).nOut(32).weightInit(WeightInit.XAVIER).activation(Activation.RELU) //First hidden layer
                            .build())
                    .layer(1, new OutputLayer.Builder().nIn(32).nOut(numRegressors).weightInit(WeightInit.XAVIER).activation(Activation.RELU) //Output layer
                            .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .build())
                    .pretrain(false).backprop(true)
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            final INDArray[] networkPredictions = new INDArray[nEpochs/ plotFrequency];
            for( int i=0; i<10; i++ ) {
                iteratorTrain.reset();
                model.fit(iteratorTrain);
            }

            System.out.println("Evaluate model....");
            RegressionEvaluation eval = model.evaluateRegression(iteratorTest);
            System.out.println(eval.stats());
            iteratorTest = new RecordReaderDataSetIterator(
                    recordReaderTest, 230, numfeatures, numRegressors);
            iteratorTest.reset();
            DataSet testData = iteratorTest.next();
            networkPredictions[0] = model.output(testData.getFeatures(), false);
            System.out.println(networkPredictions[0].toString());
            ArrayList<String> listLabels = new ArrayList<>();
            listLabels.add("prev left");
          /*  listLabels.add("prev right");
            listLabels.add("cur left");
            listLabels.add("cur right");*/
           // allData.setLabelNames(listLabels);
            //ArrayList<String> out = new ArrayList<>();
            INDArray[] yout = new INDArray[230];
            int count = 0;
            while (iteratorTest.hasNext())
            {
                yout[count] = model.output(iteratorTest, false);
                iteratorTest.next();
            }

            for (int i =0; i <= 229; i ++)
            {
             //   System.out.println(yout[i]);
            }





        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

    }


}
