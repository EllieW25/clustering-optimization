# clustering-optimization

Megh Patel, Nathan McClure, and Ellie Williams


Our project compares the runtime and efficiency of four different clustering algorithms. Our goal for this project is to get a better understanding of the differences and efficiencies of the chosen algorithms for all generated inputs. We used a synthetic data set to generate random clusters and a real data set. Using a real data set helps give a better practical understanding of each algorithm. It also makes the efficiency comparison simpler by judging how all the different algorithms handle the same set of data. The real data set we chose is based on mall shoppers. It compares their age, average salaries, and “shopping scores.” We removed the gender identities from the data set to better streamline our clusters and variables. The output are clean CSV files that show each algorithm’s runtime efficiency and can even be graphed for easier understanding.

Each algorithm was written in C++ and can be run directly in the command line. The algorithms will then make a CSV file with the run-time efficiencies of all the inputs, including the real data set. These results can either be analyzed directly or run through the included Python graphing. The correct graphing algorithm must be used for the result you intend to graph. An incorrect choice will either not run or output an incorrect graph, which can cause confusion. The graphing function is in Python; however, it can run in the command line. 




# How to Compile
Each algorithm can be compiled and ran individually using these commands:  
  &emsp;cd code\hac && g++ -o hac.exe hac.cpp  
  &emsp;cd code\k-means && g++ -o kmeans.exe k-means.cpp  
  &emsp;cd code\k-medoids && g++ -o kmedoids.exe k-medoids.cpp  
  &emsp;cd code\pso && g++ -o pso.exe pso_experiments.cpp   
  &emsp;  
  Or use the runner and compare all four algorithms at the same time (suggested but no output file is generated)  
  &emsp; cd code && g++ -o runner.exe runner.cpp

# Synthetic Dataset Generation
Compile and run our synthetic data generator  
&emsp;cd \data\synthetic && g++ -o syntheticdatagen.exe syntheticdatagen.cpp
