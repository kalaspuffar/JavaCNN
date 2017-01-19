package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;

import java.util.List;

public interface Layer {
    void forward(DataBlock db, boolean training);
    void backward();
    List<BackPropResult> getBackPropagationResult();
}
