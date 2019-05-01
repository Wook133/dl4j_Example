package Regression;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Scanner;

public class readFile {
    private static final String NEW_LINE_SEPARATOR = "\n";
    private static final String resourceDirectory = "D:\\2019\\Ex1\\src\\main\\resources\\";

    public static ArrayList<ArrayList<String>> readfile(String sfile)
    {
        ArrayList<ArrayList<String>> rawData = new ArrayList<>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(sfile));
            String line = null;
            Scanner scanner = null;
            int index = 0;
            //reader.readLine();
            while ((line = reader.readLine()) != null)
            {
                scanner = new Scanner(line);
                while (scanner.hasNext())
                {
                    String data = scanner.next();
                    String[] srow = data.split(",");
                    ArrayList<String> curline = new ArrayList<>();
                    for (String s : srow)
                    {
                        curline.add(s);
                    }
                    rawData.add(curline);
                }
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
        return rawData;
    }

    public static ArrayList<ArrayList<Double>> convertStringList(ArrayList<ArrayList<String>> rawdata)
    {
        ArrayList<ArrayList<Double>> listOut = new ArrayList<>();
        for (ArrayList<String> curList : rawdata)
        {
            ArrayList<Double> temp = new ArrayList<>();
            for (String s : curList)
            {
                Double d = Double.parseDouble(s);
                temp.add(d);
            }
            listOut.add(temp);
        }
        return listOut;
    }

    public static void displayStrings(ArrayList<ArrayList<String>> listofLists)
    {
        for (ArrayList<String> curList : listofLists)
        {
            ArrayList<Double> temp = new ArrayList<>();
            for (String s : curList)
            {
                System.out.print(s + ", ");
            }
            System.out.println();
        }
    }
    public static void displayDoubles(ArrayList<ArrayList<Double>> listofLists)
    {
        for (ArrayList<Double> curList : listofLists)
        {
            ArrayList<Double> temp = new ArrayList<>();
            for (Double d : curList)
            {
                System.out.print(d + ", ");
            }
            System.out.println();
        }
    }

    public static boolean rectangular(ArrayList<ArrayList<Double>> list)
    {
        int j = list.get(0).size();
        for (int i = 1; i <= list.size() -1; i++)
        {
            int k = list.get(i).size();
            if (j != k)
            {
                return false;
            }
        }
        return true;
    }


    public static int widthListList(ArrayList<ArrayList<Double>> list)
    {
        /*if (rectangular(list))
        {*/
            return list.get(0).size();
      /*  }
        else
            return -1;*/
    }
    public static int lengthListList(ArrayList<ArrayList<Double>> list)
    {
        return list.size();
       /* if (rectangular(list))
        {
            return list.size();
        }
        else
            return -1;*/
    }


    public static INDArray convertListList(ArrayList<ArrayList<Double>> listlist)
    {
        //The shape of the arrays are specified as integers. For example, to create a zero-filled array with 3 rows and 5 columns, use Nd4j.zeros(3,5).
        if (rectangular(listlist))
        {
            int width = widthListList(listlist);
            int height = lengthListList(listlist);
            INDArray data = Nd4j.zeros(height,width);
            //[y,x]
            for (int y = 0; y <= listlist.size() -1 ; y++) {
                for (int x = 0; x <= width - 1; x++)
                {
                    int[] index = {y,x};
                    data.putScalar(index, listlist.get(y).get(x));
                }
            }
            System.out.println("Rows    = " + data.rows());
            System.out.println("Columns = " + data.columns());
            return data;
        }
        else
            return null;
    }

    public static void main(String[] args) {
        ArrayList<ArrayList<String>> list = new ArrayList<>();
        list = readfile(resourceDirectory+"trainx.txt");
        //displayStrings(list);
        ArrayList<ArrayList<Double>> doubleList = new ArrayList<>();
        doubleList = convertStringList(list);
       // displayDoubles(doubleList);
        System.out.println(rectangular(doubleList));
        System.out.println(widthListList(doubleList));
        System.out.println(lengthListList(doubleList));
        INDArray data = convertListList(doubleList);
        System.out.println(data);
    }


}
