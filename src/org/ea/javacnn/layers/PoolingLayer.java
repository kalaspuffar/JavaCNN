package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by woden on 2017-01-19.
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
