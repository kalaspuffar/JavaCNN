package org.ea.javacnn.data;

/**
 * Holding all the data handled by the network. So a layer will receive
 * this class and return a similar block as a output that will be used
 * by the next layer in the chain.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class DataBlock {
    private int sx;
    private int sy;
    private int depth;
    private double[] w;
    private double[] dw;
}
