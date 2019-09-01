# TagEmbedSVD

This is our java implementation of the paper "TagEmbedSVD: Leveraging Tag Embeddings for Cross-Domain Collaborative Filtering".

The basic framework is inherited from Librec implementation (https://librec.net). 

## Example to Run the Codes

> cd code

> java -jar code/TagEmbedSVD.jar ../config_tagembedsvd/TagEmbedSVD.conf

Here, TagEmbedSVD.conf contains all the required configuration parameters. Alternatively, one can load the project in java IDEs such as Eclipse and run from the environment. Here, one has to make sure that path to configuration file is properly provided.

Main logic of our model can be found at the path "code/LIBREC/src/librec/rating/TagEmbedSVD.java"
