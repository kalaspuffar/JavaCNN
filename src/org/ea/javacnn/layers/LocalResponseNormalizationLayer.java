package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;

import java.util.ArrayList;
import java.util.List;

/**
 * This layer normalize the result from the convolution layer so all
 * weight values are positive, this will help the learning process and
 * shape the result.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class LocalResponseNormalizationLayer implements Layer {

    public LocalResponseNormalizationLayer(OutputDefinition def) {

    }

    @Override
    public DataBlock forward(DataBlock db, boolean training) {
        return null;
    }

    @Override
    public void backward() {

    }

    @Override
    public List<BackPropResult> getBackPropagationResult() {
        return new ArrayList<BackPropResult>();
    }
}
