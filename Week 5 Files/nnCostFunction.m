function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
 
 
X = [ones(m,1), X];
%Feedforward implementation
a1 = X;
z2 = a1*Theta1';
a2 = sigmoid(z2);
ma2 = size(a2,1);
a2 = [ones(ma2,1), a2];

z3 = a2*Theta2';
a3 = sigmoid(z3);

%Cost function implementation
y_matrix = eye(num_labels)(y,:);

T = (y_matrix .* log(a3) + (1-y_matrix) .* log(1-a3));


Jk = (-1/m)*sum(T,1);
J = sum(Jk);

%Reguarization Cost Function Implementation
Reg_Theta1 = Theta1(:,2:end);
Reg_Theta2 = Theta2(:,2:end);

Reg_Theta1_S = Reg_Theta1.^2;
Reg_Theta2_S = Reg_Theta2.^2;

T1 = sum(sum(Reg_Theta1_S));
T2 = sum(sum(Reg_Theta2_S));

J = J + ((lambda/(2*m))*(T1+T2));

%Backpropogation Implementation
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
%size(a1(1,:))
for i = 1:m,
	a1_i = a1(i,:);
	a2_i = a2(i,:);
	a3_i = a3(i,:);
	y_i = y_matrix(i, :);
	d3 = a3_i - y_i;
	d2 = (Theta2'*d3') .* sigmoidGradient([1;Theta1*a1_i']);
	Delta1 = Delta1 + d2(2:end)*a1_i;
	Delta2 = Delta2 + d3'*a2_i;
end;
	
Theta1_grad = (1/m) * Delta1 + (lambda/m)*[zeros(size(Theta1,1),1) Reg_Theta1];
Theta2_grad = (1/m) * Delta2 + (lambda/m)*[zeros(size(Theta2,1),1) Reg_Theta2];
	




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
