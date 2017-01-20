package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;

import java.util.List;

/**
 * A convolution neural network is built of layers that the data traverses back and forth in order to predict what the network sees in the data.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public interface Layer {
    void forward(DataBlock db, boolean training);
    void backward();
    List<BackPropResult> getBackPropagationResult();
}
