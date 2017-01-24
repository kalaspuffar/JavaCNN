package org.ea.javacnn.data;

/**
 * When we have done a back propagation of the network we will receive a
 * result of weight adjustments required to learn. This result set will
 * contain the data used by the trainer.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class BackPropResult {
    double l1_decay_mul, l2_decay_mul;
    private double[] w;
    private double[] dw;

    public BackPropResult(double[] w, double[] dw, double l1_decay_mul, double l2_decay_mul) {
        this.w = w;
        this.dw = dw;
        this.l1_decay_mul = l1_decay_mul;
        this.l2_decay_mul = l2_decay_mul;
    }
}
