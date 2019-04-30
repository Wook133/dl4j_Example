package Regression;

import com.opencsv.CSVReader;
import org.bytedeco.javacv.FrameFilter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.datasets.iterator.loader.DataSetLoaderIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.io.*;
import java.util.*;
import java.util.List;

public class maihn {

    private static Logger log = LoggerFactory.getLogger(maihn.class);

    private static final String NEW_LINE_SEPARATOR = "\n";

    //Random number generator seed, for reproducability
    public static final int seed = 12345;
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 1000;
    //How frequently should we plot the network output?
    public static final int plotFrequency = 5;
    //Number of data points
    public static final int nSamples = 206;
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int batchSize = 103;
    //Network learning rate
    public static final double learningRate = 0.01;
    public static final Random rng = new Random(seed);
    public static final int numInputs = 1;
    public static final int numOutputs = 1;


    public static void main(String[] args) throws Exception {


        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("xs.txt").getFile()));
        //rr.initialize(new FileSplit(new File("/path/to/myCsv.txt")));
        // Example 2: Multi-output regression from CSV, batch size 15

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int startIndex = 1;
        int numRegression = 0;
        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = nSamples;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

        DataSetIterator iterator = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
                //Specify the columns that the regression labels/targets appear in. Note that all other columns will be
                // treated as features. Columns indexes start at 0
                .regression(startIndex, startIndex+numRegression)
                .build();

        DataSet allData = iterator.next();
        allData.shuffle();
       // System.out.println(allData.toString());
        Double splitsize = 0.65;
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(splitsize);  //Use 65% of data for training (splitsize)


        recordReader.close();

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //Data Preprocessing
        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
       /* DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set
        */

        writeCsvFile("raw_trainingdata.txt", trainingData);
        writeCsvFile("raw_testdata.txt", testData);

        ArrayList<String> readList = new ArrayList<>();
        readList = readfile("raw_trainingdata.txt");
        //DisplayList(readList);
        writeConvert(readList, "trainingdata.txt");
        readList.clear();
        readList = readfile("raw_testdata.txt");
        //DisplayList(readList);
        writeConvert(readList, "testdata.txt");
        readList.clear();

        //Thread.sleep(4000);

        String resourcePath = "D:\\2019\\Ex1\\src\\main\\resources\\";

        RecordReader recordReaderTrain = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReaderTrain.initialize(new FileSplit(new File(resourcePath+"trainingdata.txt")));
        DataSetIterator iteratorTrain = new RecordReaderDataSetIterator.Builder(recordReaderTrain, batchSize)
                //Specify the columns that the regression labels/targets appear in. Note that all other columns will be
                // treated as features. Columns indexes start at 0
                .regression(startIndex, startIndex+numRegression)
                .build();
        DataSet trainData = iteratorTrain.next();


        RecordReader recordReaderTest = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReaderTest.initialize(new FileSplit(new File(resourcePath+"testdata.txt")));


        DataSetIterator iteratorTest = new RecordReaderDataSetIterator.Builder(recordReaderTest, batchSize)
                //Specify the columns that the regression labels/targets appear in. Note that all other columns will be
                // treated as features. Columns indexes start at 0
                .regression(startIndex, startIndex+numRegression)
                .build();


       // System.out.println(trainingData.toString());


       // DataSetIterator iteratorTrain = new DataSetLoaderIterator()
      //  DataSetIterator iteratorTest = new ;
        //    public ExistingDataSetIterator(@NonNull Iterable<DataSet> iterable, int totalExamples, int numFeatures, int numLabels) {

     //   System.out.println(trainingData.toString());

        final int numInputs = 4;
        int outputNum = 3;
        long seed = 6;

        final MultiLayerConfiguration conf = getShallowDenseLayerNetworkConfiguration();

        //Create the network
        final MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        //Train the network on the full data set, and evaluate in periodically
        final INDArray[] networkPredictions = new INDArray[nEpochs/ plotFrequency];
        for( int i=0; i<nEpochs; i++ ){
            //iterator.reset();
            //net.fit(trainData);
            iteratorTrain.reset();
            net.fit(iteratorTrain);
            System.out.println(i);
         /*   RegressionEvaluation regEval = net.evaluateRegression(iteratorTest);

            double mse = regEval.averageMeanSquaredError();
            double r2 = regEval.averagecorrelationR2();

          //  RegressionEvaluation regEvaluate = net.evaluateRegression(testData);
            System.out.println(net.getEpochCount() + " : " + net.score());
            System.out.println("MSE = " + mse);
            System.out.println("R2 = " + r2);
            iteratorTest.reset();*/
        }
        RegressionEvaluation regEval = net.evaluateRegression(iteratorTest);
        System.out.println(regEval.stats());
        Evaluation eval = new Evaluation(numOutputs);



       /* double mse = regEval.meanSquaredError(0);
        double r2 = regEval.correlationR2(0);
        System.out.println("MSE = " + mse);
        System.out.println("R2 = " + r2);
        List<String> prediction = new ArrayList<>();
        List<String> lbls = new ArrayList<>();
        lbls.add("x");
        lbls.add("y");
        testData.setLabelNames(lbls);
        prediction = net.predict(testData);

        System.out.println(prediction);*/
    }

    public static void main3(String[] args) throws Exception {
        //writeCsvFile("test.txt",testData);
        ArrayList<String> readList = new ArrayList<>();
        readList = readfile("test.txt");
        DisplayList(readList);
        writeConvert(readList, "testdata.txt");
    }

    /** Returns the network configuration, 2 hidden DenseLayers of size 50.
     */
    private static MultiLayerConfiguration getDeepDenseLayerNetworkConfiguration() {
        final int numHiddenNodes = 50;
        return new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.UNIFORM)
                .updater(new AdaGrad())
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.TANH).build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.TANH).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();
    }

    private static MultiLayerConfiguration getShallowDenseLayerNetworkConfiguration() {
        final int numHiddenNodes = 10;
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

    public static void writeCsvFile(String fileName, DataSet ds) throws IOException {
        FileWriter fileWriter = null;
        try
        {
            fileWriter = new FileWriter(fileName, false);
            fileWriter.append(NEW_LINE_SEPARATOR);
            fileWriter.append(ds.toString());
            fileWriter.flush();
            fileWriter.close();
        }
        catch (Exception e)
        {
            System.out.println(e);
        }
    }

    public static void writeTXTfile(String fileName, DataSet ds) throws IOException {
        FileWriter fileWriter = null;
        try
        {
            File file = new File("D:\\2019\\Ex1\\src\\main\\resources\\"+fileName);
            fileWriter = new FileWriter(file, false);
            fileWriter.append(NEW_LINE_SEPARATOR);
            fileWriter.append(ds.toString());
            fileWriter.flush();
            fileWriter.close();
        }
        catch (Exception e)
        {
            System.out.println(e);
        }
    }

    public static ArrayList<String> readFile(String s)
    {
        ArrayList<String> records = new ArrayList<>();
        try {
            CSVReader csvReader = new CSVReader(new FileReader(s));
            String[] values = null;
            String cur = "";
            while ((values = csvReader.readNext()) != null) {
                for (String st : values)
                {
                    cur = cur + st;
                }
                records.add(cur);
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();

        }
        return records;
    }

    public static ArrayList<String> readfile(String sfile)
    {
        ArrayList<String> Marks = new ArrayList<>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(sfile));
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

    //get Input
    //get Output
    // merge input and output into original format
    //write to textfile
    //read textfile to get iterator way

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

    public static void writeTF(String fileName, ArrayList<String> dataset) throws IOException {
        FileWriter fileWriter = null;
        //D:\\2019\\Ex1\\src\\main\\resources

        try
        {
            Writer output = null;
            File file = new File("D:\\2019\\Ex1\\src\\main\\resources\\"+fileName);

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

        private static String reverse(String s)
    {
        String x = "";

        for (int i = s.length() -1; i >= 0; i--)
        {
            x = x + s.charAt(i);
        }
        return x;
    }

    public static void DisplayList(ArrayList<String> list)
    {
        for (String s : list)
        {
            System.out.println(s);
        }
    }

}

