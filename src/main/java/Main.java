
import org.bytedeco.javacv.FrameFilter;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.apache.commons.lang3.math.NumberUtils.createDouble;

public class Main
{
    public static void main(String[] args) {
        try {
            System.out.println("Hello");
            int labelIndex = 4;
            int numClasses = 3;

            int batchSizeTraining = 147;
            DataSet trainingData = helper.readCSVDataset("x.csv", batchSizeTraining, labelIndex, numClasses);

// shuffle our training data to avoid any impact of ordering
            trainingData.shuffle();

            //int batchSizeTest = 3;
           // DataSet testData = readCSVDataset(irisDataTestFile, batchSizeTest, labelIndex, numClasses);
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }


        //List<List<String>> records = helper.read("x.csv");
        //List<List<Double>> out = new ArrayList<List<Double>>();
        //helper.readCSV("x.csv");

       // helper.printString(records);
    }




    public static List<List<Double>> Convert(List<List<String>> ListIn)
    {
        List<List<Double>> records = new ArrayList<List<Double>>();
        for (int i = 0; i <= ListIn.size() - 1; i++)
        {
            ArrayList<Double> currow = new ArrayList<Double>();
            for (int j = 0; j <= ListIn.get(i).size() - 1; j++)
            {
                String cur = ListIn.get(i).get(j);
                System.out.println(cur);
               // Double d = createDouble(cur);

               // currow.add(Double.valueOf(cur));
            }
            records.add(currow);
        }
        return records;
    }

}
