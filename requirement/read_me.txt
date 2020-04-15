Use conda to install python packages:

We offer three ways to install these packages on a different operating system.
* The most recommended way is to use `environment.yml` file:
    * Open terminal, run `conda env create -n name -f requirement/environment.yml python=3.6.10`.
    * If some packages cannot be installed automatically, one shall install them manually one by one.
    * If there are too many packages which cannot be installed automatically, or there are difficulties 
        to find some packages for your operating system, try the second way.
* The second recommended way is to use `install_paks.sh` file:
    * Open terminal, run `conda create --name name python=3.6.10 && conda activate name` to create and activate environment.
    * Run `./requirement/install_paks.sh` to install packages and requirements through conda.
    * If it does not work for you, you may try the third way.
* The third recommended way is to use `environment_manual.yml` file:
    * Open terminal, run `conda env create -n name -f requirement/environment_manual.yml python=3.6.10`.
    * It will only install a few important packages for you but not the entire environment, thus might cause further problems.
