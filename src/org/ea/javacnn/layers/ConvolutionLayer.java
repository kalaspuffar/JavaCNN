package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;

import java.util.ArrayList;
import java.util.List;

/**
 * This layer uses different filters to find attributes of the data that
 * affects the result. As an example there could be a filter to find
 * horizontal edges in an image.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
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
