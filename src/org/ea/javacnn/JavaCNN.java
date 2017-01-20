package org.ea.javacnn;

import org.ea.javacnn.layers.Layer;

import java.util.List;

/**
 * A network class holding the layers and some helper functions
 * for training and validation.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class JavaCNN {
    private List<Layer> layers;

    public JavaCNN(List<Layer> layers) {
        this.layers = layers;
    }
}
