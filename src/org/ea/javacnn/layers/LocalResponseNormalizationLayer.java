package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;

import java.util.ArrayList;
import java.util.List;

/**
 * This layer normalize the result from the convolution layer so all weight values are positive, this will help the learning process and shape the result.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class LocalResponseNormalizationLayer implements Layer {
    float bias_pref = 0.1f;


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
