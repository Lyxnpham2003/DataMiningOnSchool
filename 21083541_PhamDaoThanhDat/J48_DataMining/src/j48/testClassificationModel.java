package j48;

import weka.core.Instances;

public class testClassificationModel {
	public static void main(String[] args) throws Exception {
		ClassificasionModel classify= new ClassificasionModel();
		Instances data,datatrain,datatest;
		data=classify.loaddata("C:\\Program Files\\Weka-3-8-6\\data\\weather.nominal.arff");
		datatrain= classify.create_TrainTestSet(data, 70, true);
		classify.built_J48Model(datatrain);
		System.out.println(classify.outputJ48Model());
		
		datatest=classify.create_TrainTestSet(data, 70, false);
		classify.eval_J48Model(datatest);
		System.out.println(classify.outputEvalJ48Model());
		
	}
}
