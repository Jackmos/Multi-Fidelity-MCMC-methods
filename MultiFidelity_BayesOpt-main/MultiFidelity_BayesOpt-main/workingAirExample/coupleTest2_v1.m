
%using solution from https://de.mathworks.com/matlabcentral/answers/443558-matlab-crashes-when-using-conda-environment-other-than-base
pyExec = 'C:\Users\Lips\Anaconda3\envs\py310\python.exe'; %location of python version to use
pyRoot = fileparts(pyExec);
p = getenv('PATH');
p = strsplit(p, ';');
%to find the following components: from conda prompt activate environment,
%then %PATH%
addToPath = {
    pyRoot
    fullfile(pyRoot, 'Library', 'mingw-w64', 'bin')
    fullfile(pyRoot, 'Library', 'usr', 'bin')
    fullfile(pyRoot, 'Library', 'bin')
    fullfile(pyRoot, 'Scripts')
    fullfile(pyRoot, 'bin')
    };

pe = pyenv(Version=pyExec); %https://de.mathworks.com/help/matlab/matlab_external/install-supported-python-implementation.html#buialof-39


res = pyrunfile("MF_BayesOpt_Couple1.py", "goodness", fidelity_order_labels=[0,1,2,3], externalInput=1);
