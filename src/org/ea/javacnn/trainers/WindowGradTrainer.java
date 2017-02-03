package org.ea.javacnn.trainers;

import org.ea.javacnn.JavaCNN;

/**
 * This is AdaGrad but with a moving window weighted average
 * so the gradient is not accumulated over the entire history of the run.
 * it's also referred to as Idea #1 in Zeiler paper on AdaDelta.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */

public class WindowGradTrainer extends Trainer {
    private final double ro = 0.95;

    public WindowGradTrainer(JavaCNN net, int batch_size, float l2_decay) {
        super(net, batch_size, l2_decay);
    }

    @Override
    public void update(int i, int j, double gij, double[] p) {
        double[] gsumi = this.gsum.get(i);

        // this is adagrad but with a moving window weighted average
        // so the gradient is not accumulated over the entire history of the run.
        // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
        gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
        double dx = - this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij; // eps added for better conditioning
        p[j] += dx;
    }
}
