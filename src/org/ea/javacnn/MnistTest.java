package org.ea.javacnn;

import org.ea.javacnn.layers.*;
import org.ea.javacnn.trainers.SGDTrainer;
import org.ea.javacnn.trainers.Trainer;

import java.util.ArrayList;
import java.util.List;

public class MnistTest {

    public static void main(String[] argv) {
        String[] classes_txt = new String[] {"0","1","2","3","4","5","6","7","8","9"};
        String dataset_name = "mnist";
        int num_batches = 21; // 20 training batches, 1 test
        int test_batch = 20;
        int num_samples_per_batch = 3000;
        int image_dimension = 28;
        int image_channels = 1;
        boolean use_validation_data = true;
        boolean random_flip = false;
        boolean random_position = false;

        List<Layer> layers = new ArrayList<Layer>();
        layers.add(new InputLayer(24, 24, 1));
        layers.add(new ConvolutionLayer(5, 8, 1, 2));
        layers.add(new LocalResponseNormalizationLayer());
        layers.add(new PoolingLayer(2, 2));
        layers.add(new ConvolutionLayer(5, 16, 1, 2));
        layers.add(new LocalResponseNormalizationLayer());
        layers.add(new PoolingLayer(3,3));
        layers.add(new FullyConnectedLayer(10));
        layers.add(new SoftMaxLayer(10));

        JavaCNN net = new JavaCNN(layers);
        Trainer trainer = new SGDTrainer(net, "adadelta", 20, 0.001f);

    }
}
