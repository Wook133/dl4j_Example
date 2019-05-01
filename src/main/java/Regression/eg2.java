package Regression;

import org.bytedeco.javacv.FrameFilter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.loader.DataSetLoaderIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class eg2 {

    public static final int batchSize = 1460;
    public static final int nEpochs = 1000;
    public static final int numInputs = 5;
    public static final int numOutputs = 1;
    private static final String NEW_LINE_SEPARATOR = "\n";
    private static final String resourceDirectory = "D:\\2019\\Ex1\\src\\main\\resources\\";


    public static void main(String[] args) throws Exception {
        String start = "x_dx";
        String sfileend = ".txt";
        String sfile = start+sfileend;

        /*constructDataSets(start,sfile, 1,0, 206, 0.7);
        int numLinesToSkip = 0;
        char delimiter = ',';
        int startIndex = 1;
        int numRegression = 0;*/

        constructDataSets(start,sfile, 5,0, 1460, 0.7);
        int numLinesToSkip = 0;
        char delimiter = ',';
        int startIndex = 5;
        int numRegression = 0;

        RecordReader recordReaderTrain = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReaderTrain.initialize(new FileSplit(new File(resourceDirectory+start+"_trainingdata.txt")));
        DataSetIterator iteratorTrain = new RecordReaderDataSetIterator.Builder(recordReaderTrain, batchSize)
                //Specify the columns that the regression labels/targets appear in. Note that all other columns will be
                // treated as features. Columns indexes start at 0
                .regression(startIndex, startIndex+numRegression)
                .build();
        DataSet trainData = iteratorTrain.next();

        RecordReader recordReaderTest = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReaderTest.initialize(new FileSplit(new File(resourceDirectory+start+"_testdata.txt")));

        DataSetIterator iteratorTest = new RecordReaderDataSetIterator.Builder(recordReaderTest, batchSize)
                //Specify the columns that the regression labels/targets appear in. Note that all other columns will be
                // treated as features. Columns indexes start at 0
                .regression(startIndex, startIndex+numRegression)
                .build();
        DataSet testData = iteratorTest.next();
        iteratorTest.reset();


        final MultiLayerConfiguration conf = getShallowDenseLayerNetworkConfiguration();
        //Create the network
        final MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {

            net.fit(iteratorTrain);
            iteratorTrain.reset();
        }
        RegressionEvaluation regEval = net.evaluateRegression(iteratorTest);
        System.out.println(regEval.stats());


        iteratorTrain.reset();
        RegressionEvaluation regEval2 = net.evaluateRegression(iteratorTrain);
        System.out.println(regEval2.stats());
    }



    /**
     * Create Neural Network Configuration
     * @return
     */
    private static MultiLayerConfiguration getShallowDenseLayerNetworkConfiguration() {
        final int numHiddenNodes = 40;
        return new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.UNIFORM)
                .updater(new AdaGrad())
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SIGMOID)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .backprop(true)
                .pretrain(false)
                .build();
    }


    /**
     * Takes file in X,Y format
     * Converts file into Test data and Training Data
     * @param sFileName
     * @param regressionStart
     * @param predictorCount
     * @param batchSize
     * @param splitsize
     * @throws Exception
     */
    private static void constructDataSets(String start, String sFileName, int regressionStart, int predictorCount, int batchSize, Double splitsize) throws Exception
    {
        int numLinesToSkip = 0;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource(sFileName).getFile()));
        int startIndex = regressionStart;
        int numRegression = predictorCount;

        DataSetIterator iterator = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
                //Specify the columns that the regression labels/targets appear in. Note that all other columns will be
                // treated as features. Columns indexes start at 0
                .regression(startIndex, startIndex+numRegression)
                .build();

        DataSet allData = iterator.next();
        allData.shuffle();
        // System.out.println(allData.toString());
        if ((splitsize < 0) || (splitsize > 1))
        {
            splitsize = 0.8d;
        }
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(splitsize);  //Use 65% of data for training (splitsize)


        recordReader.close();

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();


        //preprocessData(trainingData, testData);

        writeTXTfile(start+"_raw_trainingdata.txt", trainingData);
        writeTXTfile(start+"_raw_testdata.txt", testData);

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set*/

        writeTXTfile(start+"_preprocessed_trainingdata.txt", trainingData);
        writeTXTfile(start+"_preprocessed_testdata.txt", testData);

        ArrayList<String> readList = new ArrayList<>();
        readList = readfile(start+"_raw_trainingdata.txt");
        //DisplayList(readList);
        writeConvert(readList, start+"_trainingdata.txt");
        readList.clear();
        readList = readfile(start+"_raw_testdata.txt");
        //DisplayList(readList);
        writeConvert(readList, start+"_testdata.txt");
        readList.clear();


    }









    private static void preprocessData(DataSet trainingData, DataSet testData, DataSet validationData)
    {
        //Data Preprocessing
        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set*/
        normalizer.transform(validationData);         //Apply normalization to the validation data. This is using statistics calculated from the *training* set*/
    }

    private static void preprocessData(DataSet trainingData, DataSet testData)
    {
        //Data Preprocessing
        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set*/
    }

    public static void writeTXTfile(String fileName, DataSet ds) throws IOException {
        FileWriter fileWriter = null;
        try
        {
            File file = new File(resourceDirectory+fileName);
            fileWriter = new FileWriter(file, false);
            fileWriter.append(NEW_LINE_SEPARATOR);
           // DataSetIterator dsi = new BaseDatasetIterator(new DataSetFetcher(ds));
            fileWriter.append(ds.toString());
            fileWriter.flush();
            fileWriter.close();
        }
        catch (Exception e)
        {
            System.out.println(e);
        }
    }

    public static void writeConvert(ArrayList<String> list, String sfile) throws Exception {
        ArrayList<String> dataset = new ArrayList<>();
        ArrayList<String> input = new ArrayList<>();
        ArrayList<String> output = new ArrayList<>();
        int firstInput = -1;
        int lastInput = -1;
        int firstOutput = -1;
        int lastOutput = -1;
        for (int i = 0; i <= list.size() - 1; i++) {
            String cur = list.get(i);
            if (cur.contains("INPUT")) {
                firstInput = i + 1;
            } else if (cur.contains("OUTPUT")) {
                lastInput = i - 1;
                firstOutput = i + 1;
            }
            /*if (cur.contains(",")) {
                cur = cur.replace(",", "");
            }*/
            if (cur.contains("]")) {
                cur = cur.replace("]", "");
            }
            if (cur.contains("[")) {
                cur = cur.replace("[", "");
            }
            dataset.add(cur);
        }
        lastOutput = list.size() - 1;
        for (int i = firstInput; i <= lastInput; i++) {
            input.add(dataset.get(i));
        }
        for (int i = firstOutput; i <= lastOutput; i++) {
            output.add(dataset.get(i));
        }
        System.out.println("input = " + input.size());
        System.out.println("output = " + output.size());
        if (input.size() == output.size())
        {
            System.out.println("BOO");
        }
        dataset.clear();
        for (int i = 0; i <= input.size() - 1; i ++)
        {
            String sin = input.get(i);
            String sout = output.get(i);
            sout = reverse(sout);
            sout =  sout.replace(",", "");
            sout = reverse(sout);
            String data = "";
            if (i == input.size() - 1)
            {
                data = sin + "," + sout;
            }
            else {
                data = sin + sout;
            }
            dataset.add(data);
        }
        writeTF(sfile, dataset);
    }

    private static String reverse(String s)
    {
        String x = "";

        for (int i = s.length() -1; i >= 0; i--)
        {
            x = x + s.charAt(i);
        }
        return x;
    }

    public static void writeTF(String fileName, ArrayList<String> dataset) throws IOException {
        FileWriter fileWriter = null;
        //D:\\2019\\Ex1\\src\\main\\resources


        try
        {
            Writer output = null;
            File file = new File(resourceDirectory+fileName);

            output = new BufferedWriter(new FileWriter(file));

            // fileWriter = new FileWriter(fileName, true);

            for (String s : dataset)
            {
                output.append(s);
                ((BufferedWriter) output).newLine();
                /*fileWriter.append(s);
                fileWriter.append(NEW_LINE_SEPARATOR);*/
            }
            output.flush();
            output.close();
            /*fileWriter.flush();
            fileWriter.close();*/
        }
        catch (Exception e)
        {
            System.out.println(e);
        }
    }

    public static ArrayList<String> readfile(String sfile)
    {
        ArrayList<String> Marks = new ArrayList<>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(resourceDirectory+sfile));
            String line = null;
            Scanner scanner = null;
            int index = 0;
            reader.readLine();
            while ((line = reader.readLine()) != null)
            {
                scanner = new Scanner(line);
                while (scanner.hasNext())
                {
                    String data = scanner.next();
                    Marks.add(data);
                }
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
        return Marks;
    }

}
