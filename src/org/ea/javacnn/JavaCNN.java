package org.ea.javacnn;

import org.ea.javacnn.layers.Layer;

import java.util.List;

/**
 *
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class JavaCNN {
    private List<Layer> layers;

    public JavaCNN(List<Layer> layers) {
        this.layers = layers;
    }
}
