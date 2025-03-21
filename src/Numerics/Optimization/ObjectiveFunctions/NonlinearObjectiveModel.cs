using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MathNet.Numerics.Optimization.ObjectiveFunctions
{
    /// <summary>
    /// Nonlinear objective model for optimization problems.
    /// Can be initialized in two ways:
    /// 1. With a model modelFunction f(x;p) and observed data (x,y) for curve fitting
    /// 2. With a direct residual modelFunction R(p) for general minimization problems
    /// </summary>
    internal class NonlinearObjectiveModel : IObjectiveModel
    {
        #region Private Variables

        /// <summary>
        /// The model modelFunction: f(x; p) that maps x to y given parameters p
        /// Null if using direct residual modelFunction mode
        /// </summary>
        readonly Func<Vector<double>, Vector<double>, Vector<double>> _modelFunction; // (p, x) => f(x; p)

        /// <summary>
        /// The derivative of model modelFunction with respect to parameters
        /// Null if using direct residual modelFunction mode or if derivative not provided
        /// </summary>
        readonly Func<Vector<double>, Vector<double>, Matrix<double>> _modelDerivative; // (p, x) => df(x; p)/dp

        /// <summary>
        /// The direct residual modelFunction: R(p) that calculates residuals directly from parameters
        /// Null if using model modelFunction mode
        /// </summary>
        readonly Func<Vector<double>, Vector<double>> _residualFunction; // p => R(p)

        /// <summary>
        /// The Jacobian of the direct residual modelFunction
        /// Null if using model modelFunction mode or if Jacobian not provided
        /// </summary>
        readonly Func<Vector<double>, Matrix<double>> _residualJacobian; // p => dR(p)/dp

        /// <summary>
        /// Flag indicating whether we're using direct residual modelFunction mode
        /// </summary>
        readonly bool _useDirectResiduals;

        readonly int _accuracyOrder; // the desired accuracy order to evaluate the jacobian by numerical approximaiton.

        Vector<double> _coefficients;

        bool _hasFunctionValue;
        double _functionValue; // the residual sum of squares, residuals * residuals.
        Vector<double> _residuals; // the weighted error values

        bool _hasJacobianValue;
        Matrix<double> _jacobianValue; // the Jacobian matrix.
        Vector<double> _gradientValue; // the Gradient vector.
        Matrix<double> _hessianValue; // the Hessian matrix.

        /// <summary>
        /// Number of observations for direct residual mode
        /// </summary>
        readonly int? _observationCount;

        #endregion Private Variables

        #region Public Variables

        /// <summary>
        /// Set or get the values of the independent variable.
        /// </summary>
        public Vector<double> ObservedX { get; private set; }

        /// <summary>
        /// Set or get the values of the observations.
        /// </summary>
        public Vector<double> ObservedY { get; private set; }

        /// <summary>
        /// Set or get the values of the weights for the observations.
        /// </summary>
        public Matrix<double> Weights { get; private set; }
        Vector<double> L; // Weights = LL'

        /// <summary>
        /// Get whether parameters are fixed or free.
        /// </summary>
        public List<bool> IsFixed { get; private set; }

        /// <summary>
        /// Get the number of observations.
        /// For direct residual mode, returns the explicitly provided observation count.
        /// If not provided and residuals are available, uses the residual vector length.
        /// </summary>
        public int NumberOfObservations
        {
            get
            {
                if (_useDirectResiduals)
                {
                    // If observation count was explicitly provided, use it
                    if (_observationCount.HasValue)
                        return _observationCount.Value;

                    // Otherwise, if we have calculated residuals, use their count
                    if (_residuals != null)
                        return _residuals.Count;

                    // If neither is available, return 0
                    return 0;
                }
                else
                {
                    return ObservedY?.Count ?? 0;
                }
            }
        }

        /// <summary>
        /// Get the number of unknown parameters.
        /// </summary>
        public int NumberOfParameters => Point?.Count ?? 0;

        /// <inheritdoc/>
        public int DegreeOfFreedom
        {
            get
            {
                var df = NumberOfObservations - NumberOfParameters;
                if (IsFixed != null)
                {
                    df = df + IsFixed.Count(p => p);
                }
                return df;
            }
        }

        /// <summary>
        /// Get the number of calls to modelFunction.
        /// </summary>
        public int FunctionEvaluations { get; set; }

        /// <summary>
        /// Get the number of calls to jacobian.
        /// </summary>
        public int JacobianEvaluations { get; set; }

        #endregion Public Variables

        /// <summary>
        /// Initializes a new instance using a model modelFunction f(x;p) for curve fitting
        /// </summary>
        /// <param name="modelFunction">The model modelFunction f(x;p) that predicts y values</param>
        /// <param name="derivative">Optional derivative modelFunction of the model</param>
        /// <param name="accuracyOrder">Accuracy order for numerical differentiation (1-6)</param>
        public NonlinearObjectiveModel(
            Func<Vector<double>, Vector<double>, Vector<double>> modelFunction,
            Func<Vector<double>, Vector<double>, Matrix<double>> derivative = null,
            int accuracyOrder = 2)
        {
            _modelFunction = modelFunction;
            _modelDerivative = derivative;
            _useDirectResiduals = false;
            _accuracyOrder = Math.Min(6, Math.Max(1, accuracyOrder));
        }

        /// <summary>
        /// Initializes a new instance using a direct residual function R(p)
        /// </summary>
        /// <param name="residualFunction">Function that directly calculates residuals from parameters</param>
        /// <param name="jacobian">Optional Jacobian of residual function</param>
        /// <param name="accuracyOrder">Accuracy order for numerical differentiation (1-6)</param>
        /// <param name="observationCount">Number of observations for degree of freedom calculation. If not provided, 
        /// will use the length of residual vector, which may not be appropriate for all statistical calculations.</param>
        public NonlinearObjectiveModel(
            Func<Vector<double>, Vector<double>> residualFunction,
            Func<Vector<double>, Matrix<double>> jacobian = null,
            int accuracyOrder = 2,
            int? observationCount = null)
        {
            _residualFunction = residualFunction ?? throw new ArgumentNullException(nameof(residualFunction));
            _residualJacobian = jacobian;
            _useDirectResiduals = true;
            _accuracyOrder = Math.Min(6, Math.Max(1, accuracyOrder));
            _observationCount = observationCount;
        }

        /// <inheritdoc/>
        public IObjectiveModel Fork()
        {
            if (_useDirectResiduals)
            {
                return new NonlinearObjectiveModel(_residualFunction, _residualJacobian, _accuracyOrder, _observationCount)
                {
                    _coefficients = _coefficients,
                    _hasFunctionValue = _hasFunctionValue,
                    _functionValue = _functionValue,
                    _residuals = _residuals,
                    _hasJacobianValue = _hasJacobianValue,
                    _jacobianValue = _jacobianValue,
                    _gradientValue = _gradientValue,
                    _hessianValue = _hessianValue,
                    IsFixed = IsFixed
                };
            }
            else
            {
                return new NonlinearObjectiveModel(_modelFunction, _modelDerivative, _accuracyOrder)
                {
                    ObservedX = ObservedX,
                    ObservedY = ObservedY,
                    Weights = Weights,
                    L = L,
                    _coefficients = _coefficients,
                    _hasFunctionValue = _hasFunctionValue,
                    _functionValue = _functionValue,
                    _residuals = _residuals,
                    _hasJacobianValue = _hasJacobianValue,
                    _jacobianValue = _jacobianValue,
                    _gradientValue = _gradientValue,
                    _hessianValue = _hessianValue,
                    IsFixed = IsFixed
                };
            }
        }

        /// <inheritdoc/>
        public IObjectiveModel CreateNew()
        {
            if (_useDirectResiduals)
            {
                return new NonlinearObjectiveModel(_residualFunction, _residualJacobian, _accuracyOrder, _observationCount);
            }
            else
            {
                return new NonlinearObjectiveModel(_modelFunction, _modelDerivative, _accuracyOrder);
            }
        }

        /// <summary>
        /// Set or get the values of the parameters.
        /// </summary>
        public Vector<double> Point => _coefficients;

        /// <summary>
        /// Get the y-values of the fitted model that correspond to the independent values.
        /// </summary>
        public Vector<double> ModelValues { get; private set; }

        /// <summary>
        /// Get the residual values at the current parameters.
        /// </summary>
        public Vector<double> Residuals
        {
            get
            {
                if (!_hasFunctionValue)
                {
                    EvaluateFunction();
                    _hasFunctionValue = true;
                }
                return _residuals;
            }
        }

        /// <inheritdoc/>
        public double Value
        {
            get
            {
                if (!_hasFunctionValue)
                {
                    EvaluateFunction();
                    _hasFunctionValue = true;
                }
                return _functionValue;
            }
        }

        /// <inheritdoc/>
        public Vector<double> Gradient
        {
            get
            {
                if (!_hasJacobianValue)
                {
                    EvaluateJacobian();
                    _hasJacobianValue = true;
                }
                return _gradientValue;
            }
        }

        /// <inheritdoc/>
        public Matrix<double> Hessian
        {
            get
            {
                if (!_hasJacobianValue)
                {
                    EvaluateJacobian();
                    _hasJacobianValue = true;
                }
                return _hessianValue;
            }
        }

        /// <inheritdoc/>
        public bool IsGradientSupported => true;

        /// <inheritdoc/>
        public bool IsHessianSupported => true;

        /// <summary>
        /// Set observed data to fit.
        /// Only applicable when using model function mode.
        /// </summary>
        public void SetObserved(Vector<double> observedX, Vector<double> observedY, Vector<double> weights = null)
        {
            if (_useDirectResiduals)
            {
                throw new InvalidOperationException("Cannot set observed data when using direct residual function mode.");
            }

            if (observedX == null || observedY == null)
            {
                throw new ArgumentNullException("The data set can't be null.");
            }
            if (observedX.Count != observedY.Count)
            {
                throw new ArgumentException("The observed x data can't have different from observed y data.");
            }
            ObservedX = observedX;
            ObservedY = observedY;

            if (weights != null && weights.Count != observedY.Count)
            {
                throw new ArgumentException("The weightings can't have different from observations.");
            }
            if (weights != null && weights.Count(x => double.IsInfinity(x) || double.IsNaN(x)) > 0)
            {
                throw new ArgumentException("The weightings are not well-defined.");
            }
            if (weights != null && weights.Count(x => x == 0) == weights.Count)
            {
                throw new ArgumentException("All the weightings can't be zero.");
            }
            if (weights != null && weights.Count(x => x < 0) > 0)
            {
                weights = weights.PointwiseAbs();
            }

            Weights = (weights == null)
                    ? null
                    : Matrix<double>.Build.DenseOfDiagonalVector(weights);

            L = (weights == null)
                ? null
                : Weights.Diagonal().PointwiseSqrt();
        }

        /// <inheritdoc/>
        public void SetParameters(Vector<double> initialGuess, List<bool> isFixed = null)
        {
            _coefficients = initialGuess ?? throw new ArgumentNullException(nameof(initialGuess));

            if (isFixed != null && isFixed.Count != initialGuess.Count)
            {
                throw new ArgumentException("The isFixed can't have different size from the initial guess.");
            }
            if (isFixed != null && isFixed.Count(p => p) == isFixed.Count)
            {
                throw new ArgumentException("All the parameters can't be fixed.");
            }
            IsFixed = isFixed;
        }

        /// <inheritdoc/>
        public void EvaluateAt(Vector<double> parameters)
        {
            if (parameters == null)
            {
                throw new ArgumentNullException(nameof(parameters));
            }
            if (parameters.Any(p => double.IsNaN(p) || double.IsInfinity(p)))
            {
                throw new ArgumentException("The parameters must be finite.");
            }

            _coefficients = parameters;
            _hasFunctionValue = false;
            _hasJacobianValue = false;

            _jacobianValue = null;
            _gradientValue = null;
            _hessianValue = null;
        }

        /// <inheritdoc/>
        public IObjectiveFunction ToObjectiveFunction()
        {
            (double, Vector<double>, Matrix<double>) Function(Vector<double> point)
            {
                EvaluateAt(point);
                return (Value, Gradient, Hessian);
            }

            var objective = new GradientHessianObjectiveFunction(Function);
            return objective;
        }

        #region Private Methods

        /// <summary>
        /// Evaluates the objective function at the current parameter values.
        /// </summary>
        void EvaluateFunction()
        {
            if (_coefficients == null)
            {
                throw new InvalidOperationException("Cannot evaluate function: current parameters is not set.");
            }

            if (_useDirectResiduals)
            {
                // Direct residual mode: calculate residuals directly from parameters
                _residuals = _residualFunction(Point);
                FunctionEvaluations++;
            }
            else
            {
                // Model function mode: calculate residuals from model predictions and observed data
                if (ModelValues == null)
                {
                    ModelValues = Vector<double>.Build.Dense(NumberOfObservations);
                }

                ModelValues = _modelFunction(Point, ObservedX);
                FunctionEvaluations++;

                // calculate the weighted residuals
                _residuals = (Weights == null)
                    ? ObservedY - ModelValues
                    : (ObservedY - ModelValues).PointwiseMultiply(L);
            }

            // Calculate the residual sum of squares with 1/2 factor
            // F(p) = 1/2 * ∑(residuals²)
            _functionValue = 0.5 * _residuals.DotProduct(_residuals);
        }

        void EvaluateJacobian()
        {
            if (_coefficients == null)
            {
                throw new InvalidOperationException("Cannot evaluate Jacobian: current parameters is not set.");
            }

            if (_useDirectResiduals)
            {
                // Direct residual mode: use provided Jacobian or calculate numerically
                if (_residualJacobian != null)
                {
                    _jacobianValue = _residualJacobian(Point);
                    JacobianEvaluations++;
                }
                else
                {
                    // Calculate Jacobian numerically for residual function
                    _jacobianValue = NumericalJacobianForResidual(Point);
                    FunctionEvaluations += _accuracyOrder * NumberOfParameters;
                }
            }
            else
            {
                // Model function mode: use provided derivative or calculate numerically
                if (_modelDerivative != null)
                {
                    // analytical jacobian
                    _jacobianValue = _modelDerivative(Point, ObservedX);
                    JacobianEvaluations++;
                }
                else
                {
                    // numerical jacobian
                    _jacobianValue = NumericalJacobian(Point, ModelValues, _accuracyOrder);
                    FunctionEvaluations += _accuracyOrder * NumberOfParameters;
                }

                // Apply weights to jacobian in model function mode
                if (Weights != null)
                {
                    for (var i = 0; i < NumberOfObservations; i++)
                    {
                        for (var j = 0; j < NumberOfParameters; j++)
                        {
                            _jacobianValue[i, j] = _jacobianValue[i, j] * L[i];
                        }
                    }
                }
            }

            // Apply fixed parameters to jacobian
            if (IsFixed != null)
            {
                for (var j = 0; j < NumberOfParameters; j++)
                {
                    if (IsFixed[j])
                    {
                        // if j-th parameter is fixed, set J[i, j] = 0
                        for (var i = 0; i < _jacobianValue.RowCount; i++)
                        {
                            _jacobianValue[i, j] = 0.0;
                        }
                    }
                }
            }

            // Gradient, g = -J'W(y − f(x; p)) = -J'L(L'E) = -J'LR
            _gradientValue = -_jacobianValue.Transpose() * _residuals;

            // approximated Hessian, H = J'WJ + ∑LRiHi ~ J'WJ near the minimum
            _hessianValue = _jacobianValue.Transpose() * _jacobianValue;
        }

        /// <summary>
        /// Calculate numerical Jacobian for model function using finite differences
        /// </summary>
        /// <param name="parameters">Current parameter values</param>
        /// <param name="currentValues">Current model values at the parameters</param>
        /// <param name="accuracyOrder">Order of accuracy for finite difference formula</param>
        /// <returns>Jacobian matrix of partial derivatives</returns>
        Matrix<double> NumericalJacobian(Vector<double> parameters, Vector<double> currentValues, int accuracyOrder = 2)
        {
            const double sqrtEpsilon = 1.4901161193847656250E-8; // sqrt(machineEpsilon)

            var derivertives = Matrix<double>.Build.Dense(NumberOfObservations, NumberOfParameters);

            var d = 0.000003 * parameters.PointwiseAbs().PointwiseMaximum(sqrtEpsilon);

            var h = Vector<double>.Build.Dense(NumberOfParameters);
            for (var j = 0; j < NumberOfParameters; j++)
            {
                h[j] = d[j];

                if (accuracyOrder >= 6)
                {
                    // f'(x) = {- f(x - 3h) + 9f(x - 2h) - 45f(x - h) + 45f(x + h) - 9f(x + 2h) + f(x + 3h)} / 60h + O(h^6)
                    var f1 = _modelFunction(parameters - 3 * h, ObservedX);
                    var f2 = _modelFunction(parameters - 2 * h, ObservedX);
                    var f3 = _modelFunction(parameters - h, ObservedX);
                    var f4 = _modelFunction(parameters + h, ObservedX);
                    var f5 = _modelFunction(parameters + 2 * h, ObservedX);
                    var f6 = _modelFunction(parameters + 3 * h, ObservedX);

                    var prime = (-f1 + 9 * f2 - 45 * f3 + 45 * f4 - 9 * f5 + f6) / (60 * h[j]);
                    derivertives.SetColumn(j, prime);
                }
                else if (accuracyOrder == 5)
                {
                    // f'(x) = {-137f(x) + 300f(x + h) - 300f(x + 2h) + 200f(x + 3h) - 75f(x + 4h) + 12f(x + 5h)} / 60h + O(h^5)
                    var f1 = currentValues;
                    var f2 = _modelFunction(parameters + h, ObservedX);
                    var f3 = _modelFunction(parameters + 2 * h, ObservedX);
                    var f4 = _modelFunction(parameters + 3 * h, ObservedX);
                    var f5 = _modelFunction(parameters + 4 * h, ObservedX);
                    var f6 = _modelFunction(parameters + 5 * h, ObservedX);

                    var prime = (-137 * f1 + 300 * f2 - 300 * f3 + 200 * f4 - 75 * f5 + 12 * f6) / (60 * h[j]);
                    derivertives.SetColumn(j, prime);
                }
                else if (accuracyOrder == 4)
                {
                    // f'(x) = {f(x - 2h) - 8f(x - h) + 8f(x + h) - f(x + 2h)} / 12h + O(h^4)
                    var f1 = _modelFunction(parameters - 2 * h, ObservedX);
                    var f2 = _modelFunction(parameters - h, ObservedX);
                    var f3 = _modelFunction(parameters + h, ObservedX);
                    var f4 = _modelFunction(parameters + 2 * h, ObservedX);

                    var prime = (f1 - 8 * f2 + 8 * f3 - f4) / (12 * h[j]);
                    derivertives.SetColumn(j, prime);
                }
                else if (accuracyOrder == 3)
                {
                    // f'(x) = {-11f(x) + 18f(x + h) - 9f(x + 2h) + 2f(x + 3h)} / 6h + O(h^3)
                    var f1 = currentValues;
                    var f2 = _modelFunction(parameters + h, ObservedX);
                    var f3 = _modelFunction(parameters + 2 * h, ObservedX);
                    var f4 = _modelFunction(parameters + 3 * h, ObservedX);

                    var prime = (-11 * f1 + 18 * f2 - 9 * f3 + 2 * f4) / (6 * h[j]);
                    derivertives.SetColumn(j, prime);
                }
                else if (accuracyOrder == 2)
                {
                    // f'(x) = {f(x + h) - f(x - h)} / 2h + O(h^2)
                    var f1 = _modelFunction(parameters + h, ObservedX);
                    var f2 = _modelFunction(parameters - h, ObservedX);

                    var prime = (f1 - f2) / (2 * h[j]);
                    derivertives.SetColumn(j, prime);
                }
                else
                {
                    // f'(x) = {- f(x) + f(x + h)} / h + O(h)
                    var f1 = currentValues;
                    var f2 = _modelFunction(parameters + h, ObservedX);

                    var prime = (-f1 + f2) / h[j];
                    derivertives.SetColumn(j, prime);
                }

                h[j] = 0;
            }

            return derivertives;
        }

        /// <summary>
        /// Calculate numerical Jacobian for direct residual function R(p)
        /// </summary>
        Matrix<double> NumericalJacobianForResidual(Vector<double> parameters)
        {
            const double sqrtEpsilon = 1.4901161193847656250E-8; // sqrt(machineEpsilon)

            // Get current residuals
            var residuals = _residualFunction(parameters);
            var residualSize = residuals.Count;

            var derivatives = Matrix<double>.Build.Dense(residualSize, NumberOfParameters);

            var d = 0.000003 * parameters.PointwiseAbs().PointwiseMaximum(sqrtEpsilon);

            var h = Vector<double>.Build.Dense(NumberOfParameters);
            for (var j = 0; j < NumberOfParameters; j++)
            {
                h[j] = d[j];

                if (_accuracyOrder >= 6)
                {
                    // f'(x) = {- f(x - 3h) + 9f(x - 2h) - 45f(x - h) + 45f(x + h) - 9f(x + 2h) + f(x + 3h)} / 60h + O(h^6)
                    var r1 = _residualFunction(parameters - 3 * h);
                    var r2 = _residualFunction(parameters - 2 * h);
                    var r3 = _residualFunction(parameters - h);
                    var r4 = _residualFunction(parameters + h);
                    var r5 = _residualFunction(parameters + 2 * h);
                    var r6 = _residualFunction(parameters + 3 * h);

                    var prime = (-r1 + 9 * r2 - 45 * r3 + 45 * r4 - 9 * r5 + r6) / (60 * h[j]);
                    derivatives.SetColumn(j, prime);
                }
                else if (_accuracyOrder == 5)
                {
                    // Implementation similar to above for 5th order accuracy
                    var r1 = residuals;
                    var r2 = _residualFunction(parameters + h);
                    var r3 = _residualFunction(parameters + 2 * h);
                    var r4 = _residualFunction(parameters + 3 * h);
                    var r5 = _residualFunction(parameters + 4 * h);
                    var r6 = _residualFunction(parameters + 5 * h);

                    var prime = (-137 * r1 + 300 * r2 - 300 * r3 + 200 * r4 - 75 * r5 + 12 * r6) / (60 * h[j]);
                    derivatives.SetColumn(j, prime);
                }
                else if (_accuracyOrder == 4)
                {
                    // Implementation similar to above for 4th order accuracy
                    var r1 = _residualFunction(parameters - 2 * h);
                    var r2 = _residualFunction(parameters - h);
                    var r3 = _residualFunction(parameters + h);
                    var r4 = _residualFunction(parameters + 2 * h);

                    var prime = (r1 - 8 * r2 + 8 * r3 - r4) / (12 * h[j]);
                    derivatives.SetColumn(j, prime);
                }
                else if (_accuracyOrder == 3)
                {
                    // Implementation similar to above for 3rd order accuracy
                    var r1 = residuals;
                    var r2 = _residualFunction(parameters + h);
                    var r3 = _residualFunction(parameters + 2 * h);
                    var r4 = _residualFunction(parameters + 3 * h);

                    var prime = (-11 * r1 + 18 * r2 - 9 * r3 + 2 * r4) / (6 * h[j]);
                    derivatives.SetColumn(j, prime);
                }
                else if (_accuracyOrder == 2)
                {
                    // f'(x) = {f(x + h) - f(x - h)} / 2h + O(h^2)
                    var r1 = _residualFunction(parameters + h);
                    var r2 = _residualFunction(parameters - h);

                    var prime = (r1 - r2) / (2 * h[j]);
                    derivatives.SetColumn(j, prime);
                }
                else
                {
                    // f'(x) = {- f(x) + f(x + h)} / h + O(h)
                    var r1 = residuals;
                    var r2 = _residualFunction(parameters + h);

                    var prime = (-r1 + r2) / h[j];
                    derivatives.SetColumn(j, prime);
                }

                h[j] = 0;
            }

            return derivatives;
        }

        #endregion Private Methods
    }
}
