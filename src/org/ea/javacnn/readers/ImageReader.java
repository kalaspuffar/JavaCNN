package org.ea.javacnn.readers;

public class ImageReader implements Reader {
    @Override
    public int readNextLabel() {
        return 0;
    }

    @Override
    public byte[] readNextImage() throws Exception {
        return new byte[0];
    }

    @Override
    public void reset() {

    }

    @Override
    public int getMaxvalue() {
        return 0;
    }

    public int numOfClasses() {
        return 0;
    }

    @Override
    public int size() {
        return 0;
    }

    @Override
    public int getSizeX() {
        return 0;
    }

    @Override
    public int getSizeY() {
        return 0;
    }
}
