%% Understanding Pytorch cross entropy function
% Softmax 
%S(Y_i) = e^Y_i/sum(e^Y_i)
% Normalizes the inputs to be between 0 and 1 but 
NeuralNetOutput = [2, 1, .1];

SoftMaxOutput = exp(NeuralNetOutput)/sum(exp(NeuralNetOutput))
labels = [1, 0, 0]

% 0.6590    0.2424    0.0986

% Cross entropy
% D(Y, y) = -1 * Sum(y_i*log(Y_i) ...Normalizing is optional
% When output of softmax does not diverge from labels, CE loss is low
CrossEntropyOutput = -1*sum(log(SoftMaxOutput).*labels)

% Example of poor soft max output that diverges from label
PoorSoftMaxOutput = [.1, .3, .6]
labels = [1, 0, 0]
HighCrossEntropyOutput = -1*sum(log(PoorSoftMaxOutput).*labels)

%% Analyzing Cross Entropy results from 2 Epochs vs 10

CELoss10Epoch = [2.223, 1.881, 1.674, 1.574, 1.492, 1.466, 1.380, 1.368, 1.348, 1.318, 1.312, 1.297, 1.207, 1.240, 1.202, 1.193, 1.183, 1.189, 1.111, 1.130, 1.104, 1.105, 1.128, 1.105, 1.033, 1.042, 1.076, 1.045, 1.053, 1.049, 0.971, 0.975, 0.981, 1.013, 1.005, 1.000, 0.916, 0.941, 0.926, 0.956, 0.968, 0.957, 0.851, 0.881, 0.905, 0.907, 0.941, 0.937, 0.812, 0.849, 0.870, 0.919, 0.895, 0.906, 0.782, 0.822, 0.855, 0.867, 0.861, 0.883]
CELoss2Epoch = [2.216, 1.882, 1.686, 1.583, 1.527, 1.499, 1.399, 1.384, 1.360, 1.337, 1.338, 1.301];
figure(1);
plot(CELoss2Epoch); title('Cross Entropy Loss After 2 Epoch: No data Augmentation')
figure(2);
plot(CELoss10Epoch); title('Cross Entropy Loss After 10 Epoch: No data Augmentation')
