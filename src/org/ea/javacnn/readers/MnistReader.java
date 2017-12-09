package org.ea.javacnn.readers;

import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

public class MnistReader implements Reader {
    private String labelFile;
    private String imageFile;

    private FileInputStream labelIO;
    private FileInputStream imageIO;

    private int labelSize;
    private int imageSize;
    private int imageX;
    private int imageY;

    private int readInt(FileInputStream is) throws Exception {
        byte[] int32Full = new byte[4];
        is.read(int32Full);
        ByteBuffer wrapped = ByteBuffer.wrap(int32Full);
        return wrapped.getInt();
    }

    @Override
    public int size() {
        return imageSize;
    }

    public MnistReader(String labelFile, String imageFile) {
        try {
            this.labelFile = labelFile;
            this.imageFile = imageFile;
            labelIO = new FileInputStream(labelFile);
            imageIO = new FileInputStream(imageFile);
            if(readInt(labelIO) != 2049) throw new Exception("Label file header missing");
            if(readInt(imageIO) != 2051) throw new Exception("Image file header missing");

            labelSize = readInt(labelIO);
            imageSize = readInt(imageIO);

            if(labelSize != imageSize) throw new Exception("Labels and images don't match in number.");

            imageY = readInt(imageIO);
            imageX = readInt(imageIO);

            System.out.println("LSZ " +labelSize + " ISZ " + imageSize + " Y " + imageY + " X " + imageX);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public int getMaxvalue() {
        return 255;
    }

    public int numOfClasses() {
        return 10;
    }

    @Override
    public void reset() {
        try {
            labelIO.close();
            imageIO.close();

            labelIO = new FileInputStream(labelFile);
            imageIO = new FileInputStream(imageFile);

            readInt(labelIO);
            readInt(labelIO);
            readInt(imageIO);
            readInt(imageIO);
            readInt(imageIO);
            readInt(imageIO);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public int readNextLabel() {
        try {
            return labelIO.read();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return -1;
    }

    @Override
    public int[] readNextImage() throws Exception {
        int size = imageX * imageY;
        byte[] imageArray = new byte[size];
        Arrays.fill(imageArray, (byte)0);
        imageIO.read(imageArray, 0, size);
        int[] imageInts = new int[size];
        for(int i=0; i<size; i++) {
            imageInts[i] = imageArray[i];
        }

        return imageInts;
    }

    public static void main(String[] argv) {
        Reader mr = new MnistReader("mnist/t10k-labels.idx1-ubyte", "mnist/t10k-images.idx3-ubyte");

        for(int i=0; i<200; i++) {
            System.out.print(mr.readNextLabel());
        }
        System.out.print(mr.readNextLabel());

        for(int i=0; i<200; i++) {
            try {
                mr.readNextImage();
            } catch (Exception e) {
                e.printStackTrace();
                System.out.println("Crash at "+i);
            }
        }

        try {
            int[] b = mr.readNextImage();
            for(int j=0; j<b.length; j++) {
                if(j % 28 == 0) System.out.println();
                System.out.print((b[j] & 0xFF) + " ");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public int getSizeX() {
        return imageX;
    }

    @Override
    public int getSizeY() {
        return imageY;
    }
}
