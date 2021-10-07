wget https://suitesparse-collection-website.herokuapp.com/MM/VLSI/vas_stokes_1M.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Goodwin/Goodwin_127.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Meng/iChem_Jacobian.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Guettel/TEM27623.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/ML_Graph/k49_norm_10NN.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/boneS01.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Bourchtein/atmosmodd.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Boeing/bcsstk36.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage13.tar.gz

for tar in *.tar.gz; do tar xvf $tar; done
