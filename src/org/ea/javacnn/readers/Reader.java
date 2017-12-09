package org.ea.javacnn.readers;

public interface Reader {
    int readNextLabel();
    byte[] readNextImage() throws Exception;
    void reset();
    int size();
    int numOfClasses();
    int getMaxvalue();
    int getSizeX();
    int getSizeY();
}
