package org.ea.javacnn;

import org.ea.javacnn.layers.*;
import org.ea.javacnn.trainers.SGDTrainer;
import org.ea.javacnn.trainers.Trainer;

import java.util.ArrayList;
import java.util.List;

/**
 * This a test network to try the network on the [Mnist dataset](https://www.google.se/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjyh9HvmNDRAhVCFCwKHV4ZAoAQFggdMAA&url=http%3A%2F%2Fyann.lecun.com%2Fexdb%2Fmnist%2F&usg=AFQjCNE_qG4M_MMtHWJVy-yfTAYhpmk0qQ&sig2=frJ3kBj3sW2pr2PxkbFmVw&bvm=bv.144224172,d.bGg)
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
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
