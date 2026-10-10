package j48;

import java.io.File;
import java.util.logging.Filter;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.instance.RemovePercentage;

public class ClassificasionModel {
DataSource source;
J48 j48;
Evaluation eval;
public Instances loaddata(String filename) throws Exception {
	source= new DataSource(filename);
	return source.getDataSet();
	}

public Instances create_TrainTestSet(Instances data ,double per, boolean isTest)throws Exception {
	RemovePercentage rm =new RemovePercentage();
	rm.setPercentage(per);
	rm.setInvertSelection(isTest);
	rm.setInputFormat(data);
	return weka.filters.Filter.useFilter(data, rm);
	
}
public void built_J48Model(Instances datatrain)throws Exception {
	datatrain.setClassIndex(datatrain.numAttributes()-1);
	j48=new J48();
	j48.buildClassifier(datatrain);
}
public void eval_J48Model(Instances datatest)throws Exception {
	datatest.setClassIndex(datatest.numAttributes()-1);
	eval= new Evaluation(datatest);
	eval.evaluateModel(j48, datatest);
}
public String outputJ48Model() {
	return j48.toString();
}
public String outputEvalJ48Model()throws Exception {
	return eval.toSummaryString()+ "\n"+ eval.toMatrixString()+"\n"+ eval.toClassDetailsString();
}

}

