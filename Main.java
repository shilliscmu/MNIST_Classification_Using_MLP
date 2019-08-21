import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class Main {

    public static void main(String[] args) {
        // write your code here
        String trainInput = args[0];
        String testInput = args[1];
        String trainOutput = args[2];
        String testOutput = args[3];
        String metricsOutput = args[4];
        int numEpochs = Integer.parseInt(args[5]);
        int hiddenUnits = Integer.parseInt(args[6]);
        int initFlag = Integer.parseInt(args[7]);
        double learningRate = Double.parseDouble(args[8]);

        try {
            Files.delete(Paths.get(trainOutput));
            Files.delete(Paths.get(testOutput));
            Files.delete(Paths.get(metricsOutput));
        } catch(IOException e) {
            System.err.print("Failed to delete: " + e.getMessage());
        }

        Pair<ArrayList<Integer>, ArrayList<ArrayList<Integer>>> trainLabelsAndFeatures = readCsv(trainInput);
        ArrayList<Integer> trainLabels = trainLabelsAndFeatures.getKey();
        ArrayList<ArrayList<Integer>> trainFeatures = trainLabelsAndFeatures.getValue();

        Pair<ArrayList<Integer>, ArrayList<ArrayList<Integer>>> testLabelsAndFeatures = readCsv(testInput);
        ArrayList<Integer> testLabels = testLabelsAndFeatures.getKey();
        ArrayList<ArrayList<Integer>> testFeatures = testLabelsAndFeatures.getValue();

        float[][] alpha = {{1,1,2,-3,0,1,-3}, {1,3,1,2,1,0,2}, {1,2,2,2,2,2,1,}, {1,1,0,2,1,-2,2}};
        float[][] beta = {{1,1,2,-2,1},{1,1,-1,1,2},{1,3,1,-1,1}};

        System.out.println("Forward Check: ");
        IntermediateQuantities fc = neuralNetworkForward(trainFeatures.get(0), trainLabels.get(0), alpha, beta);
        System.out.println("A: " + Arrays.toString(fc.getA()));
        System.out.println("Z: " + Arrays.toString(fc.getZ()));
        System.out.println("B: " + Arrays.toString(fc.getB()));
        System.out.println("Yhat: " + Arrays.toString(fc.getYhat()));
        System.out.println("J: " + fc.getJ());

        StringBuilder metrics = new StringBuilder();
        double trainCrossEntropy;
        double testCrossEntropy;
        IntermediateQuantities train;
        IntermediateQuantities test;
        ArrayList<Integer> trainPredictedLabels = new ArrayList<>();
        ArrayList<Integer> testPredictedLabels = new ArrayList<>();

        for(int e = 1; e <= numEpochs; e++) {
            for(int i = 0; i < trainFeatures.size(); i++) {
                IntermediateQuantities o = neuralNetworkForward(trainFeatures.get(i), trainLabels.get(i), alpha, beta);
                Pair<float[][], float[][]> g_alphaBeta = neuralNetworkBackward(trainFeatures.get(i), trainLabels.get(i), alpha, beta, o);
                float[][] g_alpha = g_alphaBeta.getKey();
                float[][] g_beta = g_alphaBeta.getValue();

                for(int j = 0; j < alpha.length; j++) {
                    for(int k = 0; k < alpha[j].length; k++) {
                        alpha[j][k] -= (learningRate*g_alpha[j][k]);
                    }
                }
                System.out.println("New Alpha row 1: " + Arrays.toString(alpha[0]));
                System.out.println("New Alpha row 2: " + Arrays.toString(alpha[1]));
                System.out.println("New Alpha row 3: " + Arrays.toString(alpha[2]));
                System.out.println("New Alpha row 4: " + Arrays.toString(alpha[3]));

                for(int j = 0; j < beta.length; j ++) {
                    for (int k = 0; k < beta[j].length; k++) {
                        beta[j][k] -= (learningRate*g_beta[j][k]);
                    }
                }
                System.out.println("New Beta row 1: " + Arrays.toString(beta[0]));
                System.out.println("New Beta row 2: " + Arrays.toString(beta[1]));
                System.out.println("New Beta row 3: " + Arrays.toString(beta[2]));

            }
            ArrayList<float[]> trainYhats = new ArrayList<>();
            ArrayList<float[]> testYhats = new ArrayList<>();
            for(int i = 0; i < trainFeatures.size(); i++) {
                train = neuralNetworkForward(trainFeatures.get(i), trainLabels.get(i), alpha, beta);
                System.out.println("Yhat for train example " + (i+1) + Arrays.toString(train.getYhat()));
                trainYhats.add(train.getYhat());
            }
            for(int i = 0; i < testFeatures.size(); i++) {
                test = neuralNetworkForward(testFeatures.get(i), testLabels.get(i), alpha, beta);
                testYhats.add(test.getYhat());
            }
            trainCrossEntropy = calculateMeanCrossEntropy(trainYhats, trainLabels);
            testCrossEntropy = calculateMeanCrossEntropy(testYhats, testLabels);
            metrics.append("epoch=");
            metrics.append(e);
            metrics.append(" crossentropy(train): ");
            metrics.append(trainCrossEntropy);
            metrics.append("\nepoch=");
            metrics.append(e);
            metrics.append(" crossentropy(test): ");
            metrics.append(testCrossEntropy);
            metrics.append("\n");
            trainPredictedLabels = predictMostLikelyLabels(trainYhats);
            testPredictedLabels = predictMostLikelyLabels(testYhats);
        }

        printLabels(trainPredictedLabels, trainOutput);
        printLabels(testPredictedLabels, testOutput);
        double trainError = calculateLabelingError(trainPredictedLabels, trainLabels);
        double testError = calculateLabelingError(testPredictedLabels, testLabels);
        metrics.append("error(train): ");
        metrics.append(trainError);
        metrics.append("\nerror(test): ");
        metrics.append(testError);
        printMetrics(metrics, metricsOutput);
    }

    private static IntermediateQuantities neuralNetworkForward(ArrayList<Integer> x, int y, float[][] alpha, float[][] beta) {
        float[] a = new float[alpha.length];
        for(int j = 0; j < alpha.length; j++) {
            for (int m = 0; m < alpha[j].length; m++) {
                a[j] += alpha[j][m] * x.get(m);
            }
        }

        float[] z = new float[alpha.length+1];
        z[0]=1;
        for(int j = 1; j < z.length; j++) {
            z[j] = (float)(1/(1 + Math.exp(-a[j-1])));
        }

        float[] b = new float[beta.length];
        for(int k = 0; k < beta.length; k++) {
            for(int j = 0; j < beta[k].length; j++) {
                b[k] += beta[k][j] * z[j];
            }
        }

        float[] yhat = new float[beta.length];
        double denominator = 0;
        for(int k = 0; k < beta.length; k++) {
            denominator += Math.exp(b[k]);
        }
        for(int k = 0; k < beta.length; k++) {
            yhat[k] = (float)(Math.exp(b[k]) / denominator);
        }

        double J = -1*Math.log(yhat[y]);

        return new IntermediateQuantities(x, a, z, b, yhat, J);
    }

    private static Pair<float[][], float[][]> neuralNetworkBackward(ArrayList<Integer> x, int y, float[][] alpha, float[][] beta, IntermediateQuantities o) {
        float[] g_yhat = new float[o.getYhat().length];
        Arrays.fill(g_yhat, 0);
        g_yhat[y-1] = (-1/o.getYhat()[y-1]);
        System.out.println("New g_yhat: " + Arrays.toString(g_yhat));

        float[] g_b = new float[o.getB().length];
        for(int k = 0; k < g_b.length; k++) {
            for(int l = 0; l < g_b.length; l++) {
                if (l==k) {
                    g_b[k] += (g_yhat[l]) * (o.getYhat()[l]) * (1 - o.getYhat()[k]);
                } else {
                    g_b[k] += (g_yhat[l]) * (o.getYhat()[l]) * (0 - o.getYhat()[k]);
                }
            }
        }
        System.out.println("New g_b: " + Arrays.toString(g_b));

        float[][] g_beta = new float[beta.length][beta[0].length];
        for(int k = 0; k < beta.length; k++) {
            for(int j = 0; j < beta[k].length; j++) {
                g_beta[k][j] = g_b[k]*o.getZ()[j];
            }
        }
        System.out.println("New g_beta row 1: " + Arrays.toString(g_beta[0]));
        System.out.println("New g_beta row 2: " + Arrays.toString(g_beta[1]));
        System.out.println("New g_beta row 3: " + Arrays.toString(g_beta[2]));

        float[] g_z = new float[o.getZ().length];
        for(int j = 0; j < beta[0].length; j++) { //5
            for(int k = 0; k < beta.length; k++) { //3
                g_z[j] += (g_b[k]*beta[k][j]);
            }
        }
        System.out.println("New g_z: " + Arrays.toString(g_z));

        float[] g_a = new float[o.getA().length];
        for(int j = 0; j < g_a.length; j++) {
            g_a[j] = g_z[j+1]*o.getZ()[j+1]*(1-o.getZ()[j+1]);
        }
        System.out.println("New g_a: " + Arrays.toString(g_a));

        float[][] g_alpha = new float[alpha.length][alpha[0].length];
        for(int j = 0; j < g_alpha.length; j++) {
            for(int i = 0; i < g_alpha[j].length; i++) {
                g_alpha[j][i] = g_a[j]*x.get(i);
            }
        }
        System.out.println("New g_alpha row 1: " + Arrays.toString(g_alpha[0]));
        System.out.println("New g_alpha row 2: " + Arrays.toString(g_alpha[1]));
        System.out.println("New g_alpha row 3: " + Arrays.toString(g_alpha[2]));
        System.out.println("New g_alpha row 4: " + Arrays.toString(g_alpha[3]));

        return new Pair<>(g_alpha, g_beta);
    }

    private static void printLabels(ArrayList<Integer> predictedLabels, String fileName) {
        BufferedWriter writer;
        StringBuilder sb = new StringBuilder();
        try{
            writer = new BufferedWriter(new FileWriter(fileName));
            for(int i : predictedLabels) {
                sb.append(i);
                sb.append('\n');
            }
            writer.write(sb.toString());
            writer.close();
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    private static double calculateMeanCrossEntropy(ArrayList<float[]> predictedLabels, ArrayList<Integer> labels) {
        float[] yhat;
        int y;
        double totalCrossEntropy = 0;
        for(int i = 0; i < labels.size(); i++) {
            yhat = predictedLabels.get(i);
            y = labels.get(i);
            totalCrossEntropy += -1*Math.log(yhat[y]);
        }
        return totalCrossEntropy/(double)predictedLabels.size();
    }

    private static double calculateLabelingError(ArrayList<Integer> predictedLabels, ArrayList<Integer> labels) {
        double errors = 0;
        for(int i = 0; i < labels.size(); i++) {
            int yhat = predictedLabels.get(i);
            int y = labels.get(i);
            if(yhat != y) {
                errors++;
            }
        }
        return errors / (double) predictedLabels.size();
    }

    private static ArrayList<Integer> predictMostLikelyLabels(ArrayList<float[]> yhats) {
        ArrayList<Integer> mostLikelyLabels = new ArrayList<>();
        float[] yhat;
        float highestProbableYhat = -1;
        int indexOfHighestProbableYhat = -1;
        for(int i = 0; i < yhats.size(); i++){
            yhat = yhats.get(i);
            for(int j = 0; j < yhat.length; j++) {
                if(yhat[j] > highestProbableYhat) {
                    highestProbableYhat = yhat[j];
                    indexOfHighestProbableYhat = j;
                }
            }
            mostLikelyLabels.add(i, indexOfHighestProbableYhat);
        }
        return mostLikelyLabels;
    }

    private static void printMetrics(StringBuilder metrics, String fileName) {
        BufferedWriter writer;
        try{
            writer = new BufferedWriter(new FileWriter(fileName));
            writer.write(metrics.toString());
            writer.close();
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    private static Pair<ArrayList<Integer>, ArrayList<ArrayList<Integer>>> readCsv(String fileName) {
        ArrayList<Integer> labels = new ArrayList<>();
        ArrayList<ArrayList<Integer>> features = new ArrayList<>();

        BufferedReader reader;
        String input;
        try {
            reader = new BufferedReader(new FileReader(fileName));
            while((input = reader.readLine()) != null) {
                ArrayList<Integer> exampleFeatures = new ArrayList<>();
                String[] line = input.split(",");
                labels.add(Integer.parseInt(line[0]));
                exampleFeatures.add(0, 1);
                for(int i = 1; i < line.length; i++) {
                    exampleFeatures.add(i, Integer.parseInt(line[i]));
                }
                features.add(exampleFeatures);
            }
        } catch (NullPointerException e) {
            System.err.println("Null pointer error: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("IO Error: " + e.getMessage());
        }
        return new Pair<>(labels, features);
    }

    private static class Pair<F, S> extends java.util.AbstractMap.SimpleImmutableEntry<F, S> {
        Pair(F f, S s) {
            super(f, s);
        }
    }

    private static class IntermediateQuantities {
        ArrayList<Integer> x;
        float[] a;
        float[] z;
        float[] b;
        float[] yhat;
        double J;

        public ArrayList<Integer> getX() {
            return x;
        }

        public void setX(ArrayList<Integer> x) {
            this.x = x;
        }

        public float[] getA() {
            return a;
        }

        public void setA(float[] a) {
            this.a = a;
        }

        public float[] getZ() {
            return z;
        }

        public void setZ(float[] z) {
            this.z = z;
        }

        public float[] getB() {
            return b;
        }

        public void setB(float[] b) {
            this.b = b;
        }

        float[] getYhat() {
            return yhat;
        }

        public double getJ() {
            return J;
        }

        public void setJ(double j) {
            J = j;
        }

        IntermediateQuantities(ArrayList<Integer> x, float[] a, float[] z, float[] b, float[] yhat, double J) {
            this.x = x;
            this.a = a;
            this.z = z;
            this.b = b;
            this.yhat = yhat;
            this.J = J;
        }
    }
}
