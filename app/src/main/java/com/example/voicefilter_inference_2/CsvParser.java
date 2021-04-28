package com.example.voicefilter_inference_2;

import java.io.File;
import java.io.FileNotFoundException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Scanner;

public class CsvParser {
    public ArrayList<ArrayList<Double>> parse(String path) {
        File file = new File(path);

        ArrayList<ArrayList<Double>> csv_array = new ArrayList<>();
        Scanner inputStream;

        try {
            inputStream = new Scanner(file);

            while (inputStream.hasNext()) {
                String line = inputStream.next();
                String[] values = line.split(",");
                ArrayList<Double> sub_array = new ArrayList<>();
                for (String value: values) {
                    sub_array.add(Double.parseDouble(value));
                }
                csv_array.add(sub_array);
            }

            inputStream.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        return csv_array;
    }
}
