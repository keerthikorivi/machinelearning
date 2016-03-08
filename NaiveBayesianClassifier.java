/**
 * @author keerthikorivi
 * Reference Link: http://ebiquity.umbc.edu/blogger/2010/12/07/naive-bayes-classifier-in-50-lines/
 * 
 */
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;


public class NaiveBayesianClassifier {
	private String inputTrainingFile;
	private HashMap<String, ArrayList<String>> featuresMap;
	private ArrayList<String> featureNameList;
	private HashMap<ArrayList<String>,Integer> featureCountsMap;
	private ArrayList<ArrayList<String>> featureVectorLists;
	private HashMap<String,Integer> labelCountsMap;

	public NaiveBayesianClassifier(String trainingARFFFile){
		inputTrainingFile=trainingARFFFile;
		featuresMap = new HashMap<String,ArrayList<String>>();
		featureNameList= new ArrayList<String>();
		featureCountsMap = new HashMap<ArrayList<String>,Integer>();
		featureVectorLists = new  ArrayList<ArrayList<String>>();
		labelCountsMap = new HashMap<String,Integer>();

	}

	public void readTrainingFile() throws FileNotFoundException,IOException{

		BufferedReader bufferedReader =new BufferedReader(new FileReader(inputTrainingFile));
		String line=null;
		while((line=bufferedReader.readLine())!=null){
			if(line.startsWith("%"))
				continue;
			else if(!(line.startsWith("@"))){
				ArrayList<String> inputDataList = new ArrayList<String>();
				inputDataList.addAll(Arrays.asList(line.trim().toLowerCase().split(",")));
				featureVectorLists.add(inputDataList);
			}
			else{
				if((!(line.trim().toLowerCase().startsWith("@data"))) && (!(line.toLowerCase().startsWith("@relation")))){
					featureNameList.add(line.trim().split("\\s")[1]);
					String featureMapKey=featureNameList.get(featureNameList.size()-1);
					ArrayList<String> featureMapValue= new ArrayList<String>();
					featureMapValue.addAll(Arrays.asList(line.substring((line.indexOf("{")+1), line.indexOf("}")).trim().split(",")));
					featuresMap.put(featureMapKey, featureMapValue);
				}
			}



		}
		bufferedReader.close();

	}
	public void trainNaiveBayesianClassifier(){

		//Incrementing the label count and updated feature counts map
		for(ArrayList<String> fv :featureVectorLists){
			String label=fv.get(fv.size()-1);
			Integer labelCount=new Integer(labelCountsMap.getOrDefault(label, (new Integer(0)).intValue())+1);
			labelCountsMap.put(label, labelCount);
			for(int i=0;i<fv.size()-1;i++){
				ArrayList<String> subFeatureCountListKey=new ArrayList<String>();
				subFeatureCountListKey.add(fv.get(fv.size()-1));
				subFeatureCountListKey.add(featureNameList.get(i));
				subFeatureCountListKey.add(fv.get(i));
				Integer subFeatureCountListValue = new Integer((featureCountsMap.getOrDefault(subFeatureCountListKey,(new Integer(1))).intValue())+1);
				featureCountsMap.put(subFeatureCountListKey, subFeatureCountListValue);

			}
		}

		// Increase label counts to do smoothing 
		for(String label:labelCountsMap.keySet()){
			for(int i=0;i<featureNameList.size()-1;i++){
				int updatedLabelCount=((labelCountsMap.get(label)).intValue())+((featuresMap.get(featureNameList.get(i))).size());
				labelCountsMap.put(label,new Integer(updatedLabelCount));
			}
		}

	}

	public String classifyTestData(ArrayList<String> featureVector){
		HashMap<String,Float> probabilityPerLabel = new HashMap<String,Float>();
		float maximumProbability=0;
		String labelWithMaxProbability = null;
		for(String label:labelCountsMap.keySet()){
			float logProbability=0;
			for(int i=0;i<featureVector.size()-1;i++){
				ArrayList<String> subFeatureVectorListKey = new ArrayList<String>();
				subFeatureVectorListKey.add(label);
				subFeatureVectorListKey.add(featureNameList.get(i));
				subFeatureVectorListKey.add(featureVector.get(i));
				logProbability += Math.log((featureCountsMap.getOrDefault(subFeatureVectorListKey, new Integer(1)).floatValue())/(labelCountsMap.get(label).floatValue()));

			}
			//calculating probability per label
			float sumOfLabelCounts=0;
			for(Integer i:labelCountsMap.values()){
				sumOfLabelCounts+=i.floatValue();
			}
			probabilityPerLabel.put(label,(float) (((labelCountsMap.get(label).floatValue())/sumOfLabelCounts)* (Math.exp(logProbability))));
			maximumProbability=Math.max(maximumProbability, probabilityPerLabel.get(label));
			if(probabilityPerLabel.get(label)==maximumProbability)
				labelWithMaxProbability = label;

		}
		//System.out.println(probabilityPerLabel.values());
		return labelWithMaxProbability;

	}

	public void TestClassifier(String inputTestFilePath) throws FileNotFoundException,IOException{

		BufferedReader br =new BufferedReader(new FileReader(inputTestFilePath));
		String line=null;
		int malign_lymph__malign_lymph=0;
		int malign_lymph__metastases=0;
		int metastases__malign_lymph=0;
		int metastases__metastases=0;

		while((line=br.readLine())!=null){
			if(!((line.startsWith("%"))) && (!(line.startsWith("@")))){
				ArrayList<String> inputTestDataFeatureVector = new ArrayList<String>();
				inputTestDataFeatureVector.addAll(Arrays.asList(line.trim().toLowerCase().split(",")));
				String actualLabel = inputTestDataFeatureVector.get(inputTestDataFeatureVector.size()-1);
				String predictedLabel = classifyTestData(inputTestDataFeatureVector);
				//System.out.println("Classifier: "+classifyTestData(inputTestDataFeatureVector)+" given: "+inputTestDataFeatureVector.get(inputTestDataFeatureVector.size()-1));
				if(actualLabel.equalsIgnoreCase("malign_lymph")&& predictedLabel.equalsIgnoreCase("malign_lymph"))
					malign_lymph__malign_lymph+=1;
				else if(actualLabel.equalsIgnoreCase("malign_lymph")&& predictedLabel.equalsIgnoreCase("metastases"))
					malign_lymph__metastases+=1;
				else if(actualLabel.equalsIgnoreCase("metastases")&& predictedLabel.equalsIgnoreCase("metastases"))
					metastases__metastases+=1;
				else if(actualLabel.equalsIgnoreCase("metastases")&& predictedLabel.equalsIgnoreCase("malign_lymph"))
					metastases__malign_lymph+=1;

			}
		}
		br.close();

		System.out.println("====ConfusionMatrix====");
		System.out.println("\t a \t b  <-- classified as");
		System.out.println("\t"+metastases__metastases+"\t"+metastases__malign_lymph+"  "+ "a = metastases");
		System.out.println("\t"+malign_lymph__metastases+"\t"+malign_lymph__malign_lymph+"  "+ "b = malign_lymph");
		System.out.println("");
		float totalNumberOfClassifiedInstances= malign_lymph__malign_lymph+malign_lymph__metastases+metastases__malign_lymph+metastases__metastases;
		float correctlyClassifiedInstances = malign_lymph__malign_lymph+metastases__metastases;
		float incorrectlyClassifiedInstances = metastases__malign_lymph+malign_lymph__metastases;
		System.out.println("Correctly Classifies instances: " +(int)correctlyClassifiedInstances+" CorrectnessPercentage: "+((correctlyClassifiedInstances/totalNumberOfClassifiedInstances))*100 +"%");
		System.out.println("InCorrectly Classifies instances: " + (int)incorrectlyClassifiedInstances + " InCorrectnessPercentage: "+((incorrectlyClassifiedInstances/totalNumberOfClassifiedInstances))*100+"%");

	}





	public static void main(String args[]){
		
		if(args.length!=2){
			System.out.println("Usage: <trainingDataFilePath> <testDataFilePath>");
			System.exit(-1);
		}
		
		NaiveBayesianClassifier naiveBayesianClassifier = new NaiveBayesianClassifier(args[0]);
		try {
			naiveBayesianClassifier.readTrainingFile();
			naiveBayesianClassifier.trainNaiveBayesianClassifier();
			naiveBayesianClassifier.TestClassifier(args[1]);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


	}

}
