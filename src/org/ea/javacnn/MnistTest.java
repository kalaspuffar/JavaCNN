package org.ea.javacnn;

import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;
import org.ea.javacnn.data.TrainResult;
import org.ea.javacnn.layers.*;
import org.ea.javacnn.losslayers.SoftMaxLayer;
import org.ea.javacnn.readers.ImageReader;
import org.ea.javacnn.readers.MnistReader;
import org.ea.javacnn.readers.PGMReader;
import org.ea.javacnn.readers.Reader;
import org.ea.javacnn.trainers.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This a test network to try the network on the
 * [Mnist dataset](http://yann.lecun.com/exdb/mnist/)
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class MnistTest {

    public static void main(String[] argv) {
        List<Layer> layers = new ArrayList<Layer>();
        OutputDefinition def = new OutputDefinition();

        Reader mr = new MnistReader("mnist/train-labels-idx1-ubyte", "mnist/train-images-idx3-ubyte");
//        Reader mr = new PGMReader("pgmfiles/train");
//        Reader mr = new ImageReader("pngfiles/train");

        layers.add(new InputLayer(def, mr.getSizeX(), mr.getSizeY(), 1));
        layers.add(new ConvolutionLayer(def, 5, 32, 1, 2));
        layers.add(new RectifiedLinearUnitsLayer());
        layers.add(new PoolingLayer(def, 2,2, 0));
        layers.add(new ConvolutionLayer(def, 5, 64, 1, 2));
        layers.add(new RectifiedLinearUnitsLayer());
        layers.add(new PoolingLayer(def, 2,2, 0));
        layers.add(new FullyConnectedLayer(def, 1024));
        layers.add(new LocalResponseNormalizationLayer());
        layers.add(new DropoutLayer(def));
        layers.add(new FullyConnectedLayer(def, mr.numOfClasses()));
        layers.add(new SoftMaxLayer(def));


        /*
        layers.add(new InputLayer(def, mr.getSizeX(), mr.getSizeY(), 1));
        layers.add(new FullyConnectedLayer(def, 500));
        layers.add(new SoftMaxLayer(def));
        layers.add(new FullyConnectedLayer(def, 500));
        layers.add(new SoftMaxLayer(def));
        layers.add(new FullyConnectedLayer(def, 500));
        layers.add(new SoftMaxLayer(def));
        layers.add(new DropoutLayer(def));
        layers.add(new FullyConnectedLayer(def, 2));
        layers.add(new SoftMaxLayer(def));
        */
        JavaCNN net = new JavaCNN(layers);
        Trainer trainer = new AdaGradTrainer(net, 20, 0.001f);

        Reader mrTest = new MnistReader("mnist/t10k-labels-idx1-ubyte", "mnist/t10k-images-idx3-ubyte");
        //Reader mrTest = new PGMReader("pgmfiles/test");
        //Reader mrTest = new ImageReader("pngfiles/test");

        try {
            long start = System.currentTimeMillis();

            int[] numberDistribution = new int[10];
            int[] correctPredictions = new int[10];

            TrainResult tr = null;
            DataBlock db = new DataBlock(mr.getSizeX(), mr.getSizeY(), 1, 0);
            for(int j = 1; j < 501; j++) {
                double loss = 0;
                for (int i = 0; i < mr.size(); i++) {
                    db.addImageData(mr.readNextImage(), mr.getMaxvalue());
                    tr = trainer.train(db, mr.readNextLabel());
                    loss += tr.getLoss();
                    if (i != 0 && i % 1000 == 0) {
                        System.out.println("Pass " + j + " Read images: " + i);
                        System.out.println("Training time: "+(System.currentTimeMillis() - start));
                        System.out.println("Loss: "+(loss / (double)i));
                        start = System.currentTimeMillis();
                    }
                }
                System.out.println("Loss: "+(loss / 60000.0));
                mr.reset();

                if(j != 1) {
                    System.out.println("Last run:");
                    System.out.println("=================================");
                    printPredictions(correctPredictions, numberDistribution, mrTest.size(), mrTest.numOfClasses());
                }

                start = System.currentTimeMillis();
                Arrays.fill(correctPredictions, 0);
                Arrays.fill(numberDistribution, 0);
                for(int i=0; i<mrTest.size(); i++) {
                    db.addImageData(mrTest.readNextImage(), mr.getMaxvalue());
                    net.forward(db, false);
                    int correct = mrTest.readNextLabel();
                    int prediction = net.getPrediction();
                    if(correct == prediction) correctPredictions[correct]++;
                    numberDistribution[correct]++;
                }
                mrTest.reset();
                System.out.println("Testing time: " + (System.currentTimeMillis() - start));

                System.out.println("Current run:");
                System.out.println("=================================");
                printPredictions(correctPredictions, numberDistribution, mrTest.size(), mrTest.numOfClasses());
                start = System.currentTimeMillis();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void printPredictions(int[] correctPredictions, int[] numberDistribution, int totalSize, int numOfClasses) {
        int sumCorrectPredictions = 0;
        for (int i = 0; i < numOfClasses; i++) {
            StringBuilder sb = new StringBuilder();
            sb.append("Number ");
            sb.append(i);
            sb.append(" has predictions ");
            sb.append(correctPredictions[i]);
            sb.append("/");
            sb.append(numberDistribution[i]);
            sb.append("\t\t");
            sb.append(((float) correctPredictions[i] / (float) numberDistribution[i]) * 100);
            sb.append("%");
            System.out.println(sb.toString());
            sumCorrectPredictions += correctPredictions[i];
        }
        StringBuilder sb = new StringBuilder();
        sb.append("Total correct predictions ");
        sb.append(sumCorrectPredictions);
        sb.append("/");
        sb.append(totalSize);
        sb.append("\t\t");
        sb.append(((float) sumCorrectPredictions / (float)totalSize) * 100);
        sb.append("%");
        System.out.println(sb.toString());
    }
}
