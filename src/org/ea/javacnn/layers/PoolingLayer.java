package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;

import java.util.ArrayList;
import java.util.List;

/**
 * This layer will reduce the dataset by creating a smaller zoomed out
 * version. In essence you take a cluster of pixels take the sum of them
 * and put the result in the reduced position of the new image.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class PoolingLayer implements Layer {

    private int sx;
    private int stride;

    public PoolingLayer(int sx, int stride) {
        this.sx = sx;
        this.stride = stride;
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
