import happy.coding.io.Logs;
import librec.main.LibRec;

public class librec_main {
	public static void main(String args[]) throws Exception{
			System.out.println("Recommender started");
			Logs.config("log4j.xml", true);			
			String configFile = null;
			
			//configFile = args[0];
			//configFile = "/home/vijaikumar/PROJECTS/TagEmbedSVD/dataset/report/TagEmbedSVDF_conv.conf"; //TagEmbedSVDF_sir.conf
			configFile = "/home/Studies/PROJECTS/premi/LIBREC/myconfig/TagEmbedSVD.conf";
			
			LibRec librec = new LibRec();
			librec.setConfigFiles(configFile);
			librec.execute(null);
	}
}