package org.ea.javacnn.trainers;

import org.ea.javacnn.JavaCNN;
import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.TrainResult;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

/**
 * The adaptive gradient trainer will over time sum up the square of
 * the gradient and use it to change the weights.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class AdaGradTrainer extends Trainer {

    public AdaGradTrainer(JavaCNN net, int batch_size, float l2_decay) {
        super(net, batch_size, l2_decay);
    }

    public void update(int i, int j, double gij, double[] p) {
        double[] gsumi = this.gsum.get(i);
        // adagrad update
        gsumi[j] = gsumi[j] + gij * gij;
        double dx = - this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij;
        p[j] += dx;
    }
}