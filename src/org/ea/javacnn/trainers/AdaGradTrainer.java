package org.ea.javacnn.trainers;

import org.ea.javacnn.JavaCNN;

/**
 * The adaptive gradient trainer will over time sum up the square of
 * the gradient and use it to change the weights.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class AdaGradTrainer implements Trainer {

    public AdaGradTrainer(JavaCNN net, int batch_size, float l2_decay) {

    }
}
