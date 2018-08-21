import smile.classification.SVM;
import smile.math.Math;
import smile.math.kernel.GaussianKernel;

public class GesturePredictor {
    SVM<double[]> svm = null;
    public void fit(double[][] x, int[] y, double[][] testx, int[] testy) {
        try {
            SVM<double[]> svm = new SVM<double[]>(new GaussianKernel(8.0), 5.0, Math.max(y) + 1, SVM.Multiclass.ONE_VS_ONE);
            svm.learn(x, y);
            svm.finish();

            int error = 0;
            for (int i = 0; i < testx.length; i++) {
                if (svm.predict(testx[i]) != testy[i]) {
                    error++;
                }
            }

            System.out.format("USPS error rate = %.2f%%\n", 100.0 * error / testx.length);

            System.out.println("USPS one more epoch...");
            for (int i = 0; i < x.length; i++) {
                int j = Math.randomInt(x.length);
                svm.learn(x[j], y[j]);
            }
            System.out.println("working");
            svm.finish();
            this.svm = svm;
            error = 0;
            for (int i = 0; i < testx.length; i++) {
                if (svm.predict(testx[i]) != testy[i]) {
                    error++;
                }
            }
            System.out.format("USPS error rate = %.2f%%\n", 100.0 * error / testx.length);
        } catch (Exception ex) {
            System.err.println(ex);
        }

    }

    /**
     * @param x
     * @return
     */
    public int predict(double[] x){
        return this.svm.predict(x);
    }
}
