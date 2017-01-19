package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;

import java.util.ArrayList;
import java.util.List;

/**
 */
public class ConvolutionLayer implements Layer {
    private int sx;
    private int filters;
    private int stride;
    private int padding;

    public ConvolutionLayer(int sx, int filters, int stride, int padding) {
        this.sx = sx;
        this.filters = filters;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public void forward(DataBlock db, boolean training) {

    }

    @Override
    public void backward() {

    }

    @Override
    public List<BackPropResult> getBackPropagationResult() {
        return new ArrayList<BackPropResult>();
    }
}
