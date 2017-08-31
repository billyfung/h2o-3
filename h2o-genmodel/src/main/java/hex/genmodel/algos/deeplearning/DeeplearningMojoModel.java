package hex.genmodel.algos.deeplearning;

import hex.ModelCategory;
import hex.genmodel.GenModel;
import hex.genmodel.MojoModel;
import hex.genmodel.utils.DistributionFamily;

public class DeeplearningMojoModel extends MojoModel {
  public int _mini_batch_size;
  public int _nums; // number of numerical columns
  public int _cats; // number of categorical columns
  public int[] _catoffsets;
  public double[] _normmul;
  public double[] _normsub;
  public double[] _normrespmul;
  public double[] _normrespsub;
  public boolean _use_all_factor_levels;
  public String _activation;
  public String[] _allActivations;  // store activation function of all layers
  public boolean _imputeMeans;
  public int[] _units;  // size of neural network, input, hidden layers and output layer
  public double[] _all_drop_out_ratios; // input layer and hidden layers
  public StoreWeightsBias[] _weights; // stores weights of different layers
  public StoreWeightsBias[] _bias;    // store bias of different layers
  public int[] _catNAFill; // if mean imputation is true, mode imputation for categorical columns
  public int _numLayers;    // number of neural network layers.
  public DistributionFamily _family;

  /***
   * Should set up the neuron network frame work here
   * @param columns
   * @param domains
   */
  DeeplearningMojoModel(String[] columns, String[][] domains) {
    super(columns, domains);
  }

  public void init() {
    _numLayers = _units.length-1;
    _allActivations = new String[_numLayers];
    int inputLayers = _numLayers-1;
    for (int index=0; index < (inputLayers); index++)
      _allActivations[index]=_activation;

    _allActivations[inputLayers] = this.isAutoEncoder()?_activation:(this.isClassifier()?"Softmax":"Linear");
  }

  /***
   * This method will be derived from the scoring/prediction function of deeplearning model itself.  However,
   * we followed closely what is being done in deepwater mojo.  The variable offset is not used.
   * @param dataRow
   * @param offset
   * @param preds
   * @return
   */
  @Override
  public final double[] score0(double[] dataRow, double offset, double[] preds) {
    assert(dataRow != null) : "doubles are null"; // check to make sure data is not null
    float[] input2Neurons = new float[_units[0]]; // store inputs into the neural network
/*    if (this.isAutoEncoder())
      preds = new double[_units[0]];*/
    double[] neuronsOutput;  // save output from a neural network layer
    double[] neuronsInput;    // store input to neural network layer

    // transform inputs: NAs in categoricals are always set to new extra level.
    setInput(dataRow, input2Neurons, _nums, _cats, _catoffsets, _normmul, _normsub, _use_all_factor_levels, true);
    neuronsInput = convertFloat2Double(input2Neurons);

    // proprogate inputs through neural network
    for (int layer=0; layer < _numLayers; layer++) {
      NeuralNetwork oneLayer = new NeuralNetwork(_allActivations[layer], _all_drop_out_ratios[layer], _weights[layer],
              _bias[layer], neuronsInput, _units[layer+1]);
      neuronsOutput = oneLayer.fprop1Layer();
      neuronsInput = neuronsOutput;
    }
    if (!this.isAutoEncoder())
      assert (_nclasses == neuronsInput.length) : "nclasses " + _nclasses + " neuronsOutput.length " + neuronsInput.length;
    // Correction for classification or standardize outputs
    modifyOutputs(neuronsInput, preds, dataRow);
    return preds;
  }

  public void modifyOutputs(double[] out, double[] preds, double[] dataRow) {
    if (this.isAutoEncoder()) { // only perform unscale numerical value if need
      if (_normmul!=null && _nums>0) { // undo the standardization on output
        int nodeSize = out.length-_nums;
        for (int k = 0; k<nodeSize;k++) {
          preds[k]=out[k];
        }
        for (int k=0; k<_nums; k++) {
          int offset = nodeSize+k;
          double m = _normmul==null?1.0:_normmul[k];
          double s = _normsub==null?0:_normsub[k];
          preds[offset] = out[offset]/m+s;
        }
      }
    } else {
      if (_family == DistributionFamily.modified_huber) {
        preds[0] = -1;
        preds[2] = _family.linkInv(preds[0]);
        preds[1] = 1 - preds[2];
      } else if (this.isClassifier()) {
        assert (preds.length == out.length + 1);
        for (int i = 0; i < preds.length - 1; ++i) {
          preds[i + 1] = out[i];
          if (Double.isNaN(preds[i + 1])) throw new RuntimeException("Predicted class probability NaN!");
        }

        if (_balanceClasses)
          GenModel.correctProbabilities(preds, _priorClassDistrib, _modelClassDistrib);
        preds[0] = GenModel.getPrediction(preds, _priorClassDistrib, dataRow, _defaultThreshold);
      } else {
        if (_normrespmul != null) //either both are null or none
          preds[0] = (out[0] / _normrespmul[0] + _normrespsub[0]);
        else
          preds[0] = out[0];
        // transform prediction to response space
        preds[0] = _family.linkInv(preds[0]);
        if (Double.isNaN(preds[0]))
          throw new RuntimeException("Predicted regression target NaN!");
      }
    }
  }

  public static double[] convertFloat2Double(float[] input) {
    int arraySize = input.length;
    double[] output = new double[arraySize];
    for (int index=0; index<arraySize; index++)
      output[index] = (double) input[index];
    return output;
  }

  @Override
  public double[] score0(double[] row, double[] preds) {
    return score0(row, 0.0, preds);
  }

  public int getPredsSize(ModelCategory mc) {
    return (mc == ModelCategory.AutoEncoder)? _units[0]: (isClassifier()?nclasses()+1 :2);
  }

  // class to store weight or bias for one neuron layer
  public static class StoreWeightsBias {
    double[] _wOrBValues; // store weight or bias arrays

    StoreWeightsBias(double[] values) {
      _wOrBValues = values;
    }
  }
}
