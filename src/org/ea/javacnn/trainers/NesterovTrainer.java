package org.ea.javacnn.trainers;

import org.ea.javacnn.JavaCNN;

/**
 * Created by danielp on 2/3/17.
 */
public class NesterovTrainer extends Trainer {

    public NesterovTrainer(JavaCNN net, int batch_size, float l2_decay) {
        super(net, batch_size, l2_decay);
    }

    @Override
    public void update(int i, int j, double gij, double[] p) {
        double[] gsumi = this.gsum.get(i);
        double dx = gsumi[j];
        gsumi[j] = gsumi[j] * this.momentum + this.learning_rate * gij;
        dx = this.momentum * dx - (1.0 + this.momentum) * gsumi[j];
        p[j] += dx;
    }
}
