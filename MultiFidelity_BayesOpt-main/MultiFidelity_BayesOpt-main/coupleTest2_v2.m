%set random seed
rng(42);

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

%% === setting up the Bayesian optimization ===
acqfnName = "expected-improvement"; %"expected-improvement-per-second" 
determObjective = false;

numLFs = 6; %number of low fidelity datasets
vars = [optimizableVariable('numUsedInputs', [1, numLFs], 'Type', 'integer')];
evaluationTimes = [];

for i = 1:numLFs
        vars = [vars, ...
            optimizableVariable(['fidelityX', num2str(i-1)], [0, numLFs], 'Type', 'integer'), ...
        ];
end

loss_function = @(x) loss_function_helper(x); 
xconstraint_function = @(x) xconstraint_function_helper(x);
% varNames = ["numUsedInputs", "x0fidelity", "x1fidelity", "x2fidelity", "x3fidelity"];
% initialX = array2table([99.802      91.159        42.812        99.704        99.869], 'VariableNames', varNames );

%%
tic;
results = bayesopt(loss_function, vars, ...
    "AcquisitionFunctionName",acqfnName, ...
    "XConstraintSatisfiabilitySamples",2e6, ...
    "GPActiveSetSize",2e3, ...
    "NumCoupledConstraints",0, ...
    'XConstraintFcn', xconstraint_function, ...
    "Verbose",1, ...
    "NumSeedPoints", 1, ...%    "InitialX", initialX, ...
    'IsObjectiveDeterministic', determObjective, ...
    'MaxObjectiveEvaluations', 1, ...
    'PlotFcn',{@plotObjective,@plotMinObjective, @plotElapsedTime}, ...
    'OutputFcn', {@saveToFile}, ...
    'SaveFileName', './tempres' ...
    );
evaluationTimes = [evaluationTimes; toc];
%%
for i = 1:5
    tic;
    close('all')
    contX = results.XTrace;
    contY = results.ObjectiveTrace;
    % expand the traces with the implicitly evaluated points
    if i ~= 1
        [contX, contY] = expand_traces(contX, contY, results.UserDataTrace{end,1});
    end
    
    results = bayesopt(loss_function, vars, ...
        "AcquisitionFunctionName",acqfnName, ...
        "XConstraintSatisfiabilitySamples",2e6, ...
        "GPActiveSetSize",2e3, ...
        "NumCoupledConstraints",0, ...
        'XConstraintFcn', xconstraint_function, ...
        "Verbose",1, ...
        "InitialX", contX, ...
        "InitialObjective", contY, ...
        "NumSeedPoints", 1, ...%    "InitialX", initialX, ...
        'IsObjectiveDeterministic', determObjective, ...
        'MaxObjectiveEvaluations', numel(contY) + 1, ...
        'PlotFcn',{@plotObjective,@plotMinObjective, @plotElapsedTime}, ...
        'OutputFcn', {@saveToFile}, ...
        'SaveFileName', './tempres' ...
        );
    evaluationTimes = [evaluationTimes; toc];

end
%%
[contX, contY] = expand_traces(contX, contY, results.UserDataTrace{end,1});
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = ['simResults_', timestamp, '.mat'];
save(filename, 'contX', 'contY', 'evaluationTimes')
%%
%condition: maximally use X inputs: this should result from the
%hierarchical structure, not from putting a condition in place!
mask = contX.numUsedInputs <= 1e6;
[~, argmin] = min(contY(mask));
maskedContX = contX(mask,:);
maskedContX{argmin,:}
%% functions

function [objective,coupledconstraints, userdata] = loss_function_helper(x)
    % [numUsedInputs, x0fid, x1fid, x2fid, x3fid] = [2 0 1 3 4] should give [1 0]
    % as input, if [2 3 1 0 4] it should give [1 2] as input: it should give
    % the indices of the columns -1 (because matlab starts at 1 and python
    % at 0) of the columns with an entry < numUsedInputs and in order so that the
    % highest fidelity (that is with entry 0) is always at the end
    fidelityRanking = table2array(removevars(x, 'numUsedInputs'));
    disp(fidelityRanking)
    [sorted, sortedIndices] = sort(fidelityRanking);
    sensorOrder = sortedIndices-1;
    % numUsedInputs: only keep inputs which were ranked beloy the numUsedInputs
    sensorOrder = sensorOrder(sorted < x.('numUsedInputs'));
    % now invert this: the one with the lowest value has the highest
    % fidelity and should be last!
    sensorOrder = flip(sensorOrder);
    %disp(x)
    [last_goodness, all_goodness] = pyrunfile("MF_BayesOpt_Couple1.py", ["goodness" "all_goodness_val"], fidelity_order_labels=sensorOrder, externalInput=1);
    objective = 1; %placekeeper, calculated point is deleted anyway
    coupledconstraints = [];
    userdata = {sensorOrder, all_goodness};
end

function feasible = xconstraint_function_helper(XTable)
    % only feasible if the row is a permutation (that is, if each number is
    % only used once) only feasible points are permutations
    % -> sort the rows and check if they are the same as 0:numel, then take
    % the sum and compare with numel (basically check if complete row is
    % the same)
    [sorted, sortedIndices] = sort(table2array(XTable),2);
    feasible = ismember(sorted,0:(size(sorted,2)-1),'rows');
end

function [contX, contY] = expand_traces(contX, contY, lastUserData)
    % takes the last user data and processes adds all intermediately
    % evaluated points, as well as all permutations that would lead to the
    % same results to the traces
    
    % first remove the last row from contX and contY
    contX = contX(1:(end-1),:);
    contY = contY(1:(end-1));

    % extract the used inputs and all_goodness from the userData
    fullSensorOrder = lastUserData{1, 1};
    all_goodness = double(lastUserData{1, 2});
    numTotalInputs = size(contX,2) - 1;

    % now iterate over each evaluated combination and look at all
    % permutations that would lead to the same result
    for i = 1:numel(fullSensorOrder)
        goodness = all_goodness(i);
        numUsedInputs = i; %numUsedInputs = number of sensors or datasets being used
        sensorOrder = fullSensorOrder(1:i);
        sensorOrder = flip(sensorOrder); %now sensorOrder again so that highest fidelity one is first instead of last
        sensorRank = 0:(i-1); %corresponding rank
        %number of permutations = number of rows we need = number of
        %elements that are not given as inputs
        numUnusedSensors =  numTotalInputs - numUsedInputs;
        numRows = factorial(numUnusedSensors);
        unusedSensorRank = i:(numTotalInputs - 1);
        unusedSensorRank = unusedSensorRank + 1; %need to get +1 because the numUsedInputs will have the in between number
        allPerms = perms(unusedSensorRank);
        
        variableNames = contX.Properties.VariableNames;
        variableTypes = repmat({'double'}, [1, numel(variableNames)]);
        newX = table('Size', [numRows, numel(variableNames)], 'VariableNames', variableNames, 'VariableTypes', variableTypes);
        newX{:,1} = numUsedInputs*ones([numRows 1]);
        k = 1; %next column of allPerms that has not been used yet
        for j = 1:numTotalInputs
            if ismember(j,sensorOrder+1)
                newX{:,j+1} = sensorRank(sensorOrder+1==j)*ones([numRows 1]);
            else
                newX{:,j+1} = allPerms(:,k);
                k = k+1;
            end
        end
    contX = [contX; newX];
    contY = [contY; goodness*ones([numRows 1])];

    end
end

%% notes
%note: useless signals can either be discarded or used at the very end. If
%there were a small penalty for number of variables used the discarding
%would be more common, else we need to manually check after the
%optimization what the improvement of the objective is with added steps and
%add this knowledge. If we were to not only output the evaluated points but
%also the intermediate models, we would converge a lot quicker!! Then we
%would have to rebuild the model every time manually taking into account
%all previous X AND additional X we implicitly evaluated (outputing also
%additional stuff in the objective as user data) and the per-second should
%then go.evaluation time would then be proportional to knowledge gain
%(those with more stages take longer to evaluate but also give more
%insight, therefore balancing out the additional cost).

% still better would be if we would check if we already tried a combination
% that could be used to not start from 0 in the evaluation of the objective
% function( for example if we try [0 1 2] and [0 1 3] that we start from [0
% 1] if we already calculated it. This would require paolo to store results
% and retrieve them from a pickle or sth.

