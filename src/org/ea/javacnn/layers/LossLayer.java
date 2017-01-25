package org.ea.javacnn.layers;

/**
 * Created by danielp on 1/25/17.
 */
public abstract class LossLayer implements Layer {
    @Override
    public void backward() {}
    public abstract double backward(int y);
}
