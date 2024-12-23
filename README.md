MDFGNN-SMMA: Prediction of Potential Small Molecule-miRNA Associations Based on Multi-source Data Fusion and Graph Neural Networks

 In this study, we proposed a deep learning method called Multi-source Data Fusion and Graph Neural Networks for Small Molecule-MiRNA Association (MDFGNN-SMMA) to predict potential SM-miRNA associations. Firstly, MDFGNN-SMMA extracted features of Atom Pairs fingerprints and Molecular ACCess System fingerprints to derive fusion feature vectors for small molecules (SMs). The K-mer features were employed to generate the initial feature vectors for miRNAs. Secondly, cosine similarity measures were computed to construct the adjacency matrices for SMs and miRNAs, respectively. Thirdly, these feature vectors and adjacency matrices were input into a model comprising GAT and GraphSAGE, which were utilized to generate the final feature vectors for SMs and miRNAs. Finally, the averaged final feature vectors were utilized as input for a multi-layer perceptron to predict the associations between SMs and miRNAs. 

 Default parameter settings of MDFGNN-SMMA:
Epochs	120
Patience	5
The learning rate	1.00E-04
Number of batch size	128
Number of hidden units	32
Number of head attentions	2
Alpha for the Leaky ReLU	0.2
Dropout of GraphSAGE	0.5

Related packages：
_libgcc_mutex             0.1                       
ase                       3.22.1                  
ca-certificates           2024.3.11            
certifi                   2022.12.7      
charset-normalizer        3.3.2                     
cycler                    0.11.0                  
fonttools                 4.38.0                  
googledrivedownloader     0.4                       
h5py                      3.8.0                   
idna                      3.7                      
importlib-metadata        4.13.0                   
isodate                   0.6.1                    
Jinja2                    3.1.3                 
joblib                    1.3.2                    
kiwisolver                1.4.5                   
ld_impl_linux-64          2.38                 
libffi                    3.3                
libgcc-ng                 9.1.0               
libstdcxx-ng              9.1.0              
llvmlite                  0.39.1               
MarkupSafe                2.1.5                   
matplotlib                3.5.3                   
ncurses                   6.3                
networkx                  2.6.3                    
numba                     0.56.4                   
numpy                     1.21.6                    
openssl                   1.1.1w             
packaging                 24.0                     
pandas                    1.3.5                    
Pillow                    9.5.0                     
pip                       22.3.1          
plyfile                   0.9                      
psutil                    5.9.8                    
pyparsing                 3.1.2                    
python                    3.7.13               
python-dateutil           2.9.0.post0               
pytz                      2024.1                   
PyYAML                    6.0.1                    
rdflib                    6.3.2                   
readline                  8.1.2               
requests                  2.31.0                   
scikit-learn              1.0.2                     
scipy                     1.7.3                   
setuptools                65.6.3          
six                       1.16.0                   
sqlite                    3.38.5              
threadpoolctl             3.1.0                     
tk                        8.6.12             
torch                     1.10.1+cu111              
torch-cluster             1.5.9                    
torch-geometric           2.0.4                    
torch-scatter             2.0.9                    
torch-sparse              0.6.12                   
torchaudio                0.10.1+cu111             
torchvision               0.11.2+cu111            
tqdm                      4.66.2                   
typing_extensions         4.7.1                     
urllib3                   2.0.7                     
wheel                     0.38.4           
xz                        5.2.5                
yacs                      0.1.8                     
zipp                      3.15.0                  
zlib                      1.2.12              

After installing the required environment and packages, run‘python main.py’ command
