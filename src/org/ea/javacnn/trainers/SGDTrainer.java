package org.ea.javacnn.trainers;

import org.ea.javacnn.JavaCNN;

/**
 * Created by danielp on 2/3/17.
 */
public class SGDTrainer extends Trainer {

    public SGDTrainer(JavaCNN net, int batch_size, float l2_decay) {
        super(net, batch_size, l2_decay);
    }

    @Override
    public void update(int i, int j, double gij, double[] p) {
        double[] gsumi = this.gsum.get(i);
        // assume SGD
        if(this.momentum > 0.0) {
            // momentum update
            double dx = this.momentum * gsumi[j] - this.learning_rate * gij; // step
            gsumi[j] = dx; // back this up for next iteration of momentum
            p[j] += dx; // apply corrected gradient
        } else {
            // vanilla sgd
            p[j] +=  - this.learning_rate * gij;
        }
    }
}
