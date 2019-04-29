import com.opencsv.CSVReader;
import org.bytedeco.javacv.FrameFilter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;


public class helper
{
    private static final String NEW_LINE_SEPARATOR = "\n";
    public static INDArray readFile(String sfile)
    {
        //Nd4j.create(double[])
        ArrayList<ArrayList<Double>> list = new ArrayList<ArrayList<Double>>();
        INDArray out = null;
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
                    ArrayList<Double> row = new ArrayList<Double>();
                    String data = scanner.next();
                    if (index == 0)
                    {
                        String[] srow = data.split(",");
                        double[] drow = new double[srow.length];
                        for (int i = 0; i <= srow.length - 1; i++) {
                            drow[i] = Double.valueOf(srow[i]);
                        }
                        out = Nd4j.create(drow);
                        index=index+1;
                    }
                    else {
                        String[] srow = data.split(",");
                        double[] drow = new double[srow.length];
                        for (int i = 0; i <= srow.length - 1; i++) {
                            drow[i] = Double.valueOf(srow[i]);
                        }
                        INDArray temp = Nd4j.create(drow);
                        INDArray temp2 = Nd4j.vstack(out, temp);
                        out = temp2.dup();
                    }
                    //System.out.println();
                   // list.add(row);
                }
            }

            reader.close();
            return out;
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }


        return Nd4j.zeros(5,5);
    }

    public static  INDArray readCSV(String sFileName)
    {
        // 2. np.genfromtxt('file.csv',delimiter=',') - From a CSV file
        INDArray readFromCSV = null;
        try {
            //readFromCSV = Nd4j.readNumpy(makeResourcePath("/numpy_cheatsheet/file.csv"), ",");
            readFromCSV = Nd4j.readNumpy(makeResourcePath("java/x.csv"), ",");
            System.out.println("Read from csv readFromCSV");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return readFromCSV;
    }
    private static String makeResourcePath(String template) {
        return helper.class.getResource(template).getPath();
    }

    //https://www.opencodez.com/java/deeplearaning4j.htm
    public static DataSet readCSVDataset(String csvFileClasspath, int batchSize, int labelIndex, int numClasses)
            throws IOException, InterruptedException {

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
        return iterator.next();
    }


    public static List<List<String>> read(String s)
    {
        List<List<String>> records = new ArrayList<List<String>>();
        try {
            CSVReader csvReader = new CSVReader(new FileReader(s));
            String[] values = null;
            while ((values = csvReader.readNext()) != null) {
                records.add(Arrays.asList(values));
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();

        }
        return records;
    }

    public static void printDouble(List<List<Double>> ListIn)
    {
        for (int i = 0; i <= ListIn.size() - 1; i ++)
        {
            for (int j = 0; j <= ListIn.get(i).size() - 1; j++)
            {
                Double cur = ListIn.get(i).get(j);
                System.out.print(cur + ", ");
            }
            System.out.println();
        }
    }
    public static void printString(List<List<String>> ListIn)
    {
        for (int i = 0; i <= ListIn.size() - 1; i ++)
        {
            for (int j = 0; j <= ListIn.get(i).size() - 1; j++)
            {
                String cur = ListIn.get(i).get(j);
                System.out.print(cur + " ");
            }
            System.out.println();
        }
    }

}
