package org.ea.javacnn.data;

/**
 * Created by danielp on 1/27/17.
 */
public class TrainResult {

    long fwd_time, bwd_time;
    double l2_decay_loss, l1_decay_loss, cost_loss, softmax_loss, loss;

    public TrainResult(long fwd_time, long bwd_time, double l1_decay_loss, double l2_decay_loss, double cost_loss, double softmax_loss, double loss) {
        this.fwd_time = fwd_time;
        this.bwd_time = bwd_time;
        this.l1_decay_loss = l1_decay_loss;
        this.l2_decay_loss = l2_decay_loss;
        this.cost_loss = cost_loss;
        this.softmax_loss = softmax_loss;
        this.loss = loss;
    }
}
