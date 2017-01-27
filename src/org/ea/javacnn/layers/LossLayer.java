package org.ea.javacnn.layers;

import org.ea.javacnn.data.DataBlock;

/**
 * Created by danielp on 1/25/17.
 */
public abstract class LossLayer implements Layer {
    @Override
    public void backward() {}
    public abstract double backward(int y);
    public abstract DataBlock getOutAct();
}
