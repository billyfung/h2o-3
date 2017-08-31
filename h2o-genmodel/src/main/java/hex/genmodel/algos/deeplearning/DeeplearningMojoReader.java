package hex.genmodel.algos.deeplearning;

import hex.genmodel.ModelMojoReader;
import hex.genmodel.utils.DistributionFamily;

import java.io.IOException;

public class DeeplearningMojoReader extends ModelMojoReader<DeeplearningMojoModel> {

  @Override
  public String getModelName() {
    return "Deep Learning";
  }

  @Override
  protected void readModelData() throws IOException {
/*    if (_model.isAutoEncoder()) {
      throw new UnsupportedOperationException("AutoEncoder mojo is not ready for deployment.  Stay tuned...");
    }*/
    _model._mini_batch_size=readkv("mini_batch_size");
    _model._nums = readkv("nums");
    _model._cats = readkv("cats");
    _model._catoffsets = readkv("cat_offsets", new int[0]);
    _model._normmul = readkv("norm_mul", new double[0]);
    _model._normsub = readkv("norm_sub", new double[0]);
    _model._normrespmul = readkv("norm_resp_mul");
    _model._normrespsub = readkv("norm_resp_sub");
    _model._use_all_factor_levels = readkv("use_all_factor_levels");
    _model._activation = readkv("activation");
    _model._imputeMeans = readkv("mean_imputation");
    _model._family = DistributionFamily.valueOf((String)readkv("distribution"));
    if (_model._imputeMeans & (_model._cats > 0)) {
      _model._catNAFill = readkv("cat_modes", new int[0]);
    }
    _model._units = readkv("neural_network_sizes", new int[0]);
    _model._all_drop_out_ratios = readkv("hidden_dropout_ratios", new double[0]);

    // read in biases and weights for each layer
    int numLayers = _model._units.length-1; // exclude the output nodes.
    _model._bias = new DeeplearningMojoModel.StoreWeightsBias[numLayers];
    _model._weights = new DeeplearningMojoModel.StoreWeightsBias[numLayers];
    for (int layerIndex = 0; layerIndex < numLayers; layerIndex++) {
      double[] temp = readkv("bias_layer"+layerIndex, new double[0]);
      _model._bias[layerIndex] = new DeeplearningMojoModel.StoreWeightsBias(temp);
      temp = readkv("weight_layer"+layerIndex, new double[0]);
      _model._weights[layerIndex] = new DeeplearningMojoModel.StoreWeightsBias(temp);
    }

    _model.init();
  }

  @Override
  protected DeeplearningMojoModel makeModel(String[] columns, String[][] domains) {
    return new DeeplearningMojoModel(columns, domains);
  }
}
