function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h2=sigmoid(X*theta);
dummy2=-y'*log(h2)-(1-y')*log(1-h2);
J=((sum(dummy2)/m)+((sum(theta(2:length(theta)).^2))*lambda/(2*m)));
dummy3=(X'*(h2-y))/m;
grad=dummy3+(lambda*(theta))/m;
grad(1)=dummy3(1);




% =============================================================

end
