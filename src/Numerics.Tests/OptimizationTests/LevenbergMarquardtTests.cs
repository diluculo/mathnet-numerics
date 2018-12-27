using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Optimization;
using MathNet.Numerics.UnitTests.OptimizationTests.TestFunctions;
using NUnit.Framework;
using NUnit.Framework.Interfaces;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MathNet.Numerics.UnitTests.OptimizationTests
{
    [TestFixture]
    public class LevenbergMarquardtTests
    {
        // model: Rosenbrock
        //       f(x; a, b) = (1 - a)^2 + 100*(b - a^2)^2
        // derivatives:
        //       df/da = 400*a^3 - 400*a*b + 2*a - 2
        //       df/db = 200*(b - a^2)
        // best fitted parameters:
        //       a = 1
        //       b = 1
        private Vector<double> RosenbrockFunction(Vector<double> x, Vector<double> p)
        {
            var y = Vector<double>.Build.Dense(x.Count);
            for (int i = 0; i < x.Count; i++)
            {
                y[i] = Math.Pow(1 - p[0], 2) + 100 * Math.Pow(p[1] - p[0] * p[0], 2);
            }
            return y;
        }
        private Matrix<double> RosenbrockPrime(Vector<double> x, Vector<double> p)
        {
            var prime = Matrix<double>.Build.Dense(x.Count, p.Count);
            for (int i = 0; i < x.Count; i++)
            {
                prime[i, 0] = 400 * p[0] * p[0] * p[0] - 400 * p[0] * p[1] + 2 * p[0] - 2;
                prime[i, 1] = 200 * (p[1] - p[0] * p[0]);
            }
            return prime;
        }
        private Vector<double> Rosenbrock_x = Vector<double>.Build.Dense(2);
        private Vector<double> Rosenbrock_y = Vector<double>.Build.Dense(2);
        private Vector<double> Rosenbrock_p = Vector<double>.Build.DenseOfArray(new double[2] { 1.0, 1.0 });

        [Test]
        public void LMDER_FindMinimum_Rosenbrock()
        {
            var obj = ObjectiveFunction.Jacobian(RosenbrockFunction, RosenbrockPrime, Rosenbrock_x, Rosenbrock_y);
            var solver = new LevenbergMarquardtMinimizer(maximumIterations: 10000);
            var initialGuess = new DenseVector(new[] { -1.2, 1.0 });

            var result = solver.FindMinimum(obj, initialGuess);

            AssertHelpers.AlmostEqualRelative(Rosenbrock_p[0], result.MinimizingPoint[0], 3);
            AssertHelpers.AlmostEqualRelative(Rosenbrock_p[1], result.MinimizingPoint[1], 3);
        }

        [Test]
        public void LMDIF_FindMinimum_Rosenbrock()
        {
            var obj = ObjectiveFunction.Jacobian(RosenbrockFunction, Rosenbrock_x, Rosenbrock_y, accuracyOrder:6);
            var solver = new LevenbergMarquardtMinimizer(maximumIterations: 10000);
            var initialGuess = new DenseVector(new[] { -1.2, 1.0 });

            var result = solver.FindMinimum(obj, initialGuess);
            
            AssertHelpers.AlmostEqualRelative(Rosenbrock_p[0], result.MinimizingPoint[0], 3);
            AssertHelpers.AlmostEqualRelative(Rosenbrock_p[1], result.MinimizingPoint[1], 3);
        }

        // model: BoxBod (https://www.itl.nist.gov/div898/strd/nls/data/boxbod.shtml)
        //       f(x; a, b) = a*(1 - exp(-b*x))
        // derivatives:
        //       df/da = 1 - exp(-b*x)
        //       df/db = a*x*exp(-b*x)
        // best fitted parameters:
        //       a = 2.1380940889E+02 +/- 1.2354515176E+01
        //       b = 5.4723748542E-01 +/- 1.0455993237E-01
        private Vector<double> BoxBodFunction(Vector<double> x, Vector<double> p)
        {
            var y = Vector<double>.Build.Dense(x.Count);
            for (int i = 0; i < x.Count; i++)
            {
                y[i] = p[0] * (1 - Math.Exp(-p[1] * x[i]));
            }
            return y;
        }
        private Matrix<double> BoxBodPrime(Vector<double> x, Vector<double> p)
        {
            var prime = Matrix<double>.Build.Dense(x.Count, p.Count);
            for (int i = 0; i < x.Count; i++)
            {
                prime[i, 0] = 1 - Math.Exp(-p[1] * x[i]);
                prime[i, 1] = p[0] * x[i] * Math.Exp(-p[1] * x[i]);
            }
            return prime;
        }
        private Vector<double> BoxBod_x = Vector<double>.Build.DenseOfArray(new double[6] { 1, 2, 3, 5, 7, 10 });
        private Vector<double> BoxBod_y = Vector<double>.Build.DenseOfArray(new double[6] { 109, 149, 149, 191, 213, 224 });
        private Vector<double> BoxBod_p = Vector<double>.Build.DenseOfArray(new double[2] { 2.1380940889E+02, 5.4723748542E-01 });

        [Test]
        public void LMDER_FindMinimum_BoxBod_Unconstrained()
        {
            var obj = ObjectiveFunction.Jacobian(BoxBodFunction, BoxBodPrime, BoxBod_x, BoxBod_y);
            var solver = new LevenbergMarquardtMinimizer();
            var initialGuess = new DenseVector(new[] { 1.0, 1.0 });

            var result = solver.FindMinimum(obj, initialGuess);

            AssertHelpers.AlmostEqualRelative(BoxBod_p[0], result.MinimizingPoint[0], 6);
            AssertHelpers.AlmostEqualRelative(BoxBod_p[1], result.MinimizingPoint[1], 6);
        }

        [Test]
        public void LMDIF_FindMinimum_BoxBod_Unconstrained()
        {
            var obj = ObjectiveFunction.Jacobian(BoxBodFunction, BoxBod_x, BoxBod_y, accuracyOrder:6);
            var solver = new LevenbergMarquardtMinimizer();
            var initialGuess = new DenseVector(new[] { 1.0, 1.0 });

            var result = solver.FindMinimum(obj, initialGuess);

            AssertHelpers.AlmostEqualRelative(BoxBod_p[0], result.MinimizingPoint[0], 6);
            AssertHelpers.AlmostEqualRelative(BoxBod_p[1], result.MinimizingPoint[1], 6);
        }

        // model : Thurber (https://www.itl.nist.gov/div898/strd/nls/data/thurber.shtml)
        //       f(x; b1 ... b7) = (b1 + b2*x + b3*x^2 + b4*x^3) / (1 + b5*x + b6*x^2 + b7*x^3) 
        // derivatives:
        //       df/db1 = 1/(b5*x + b6*x^2 + b7*x^3 + 1)
        //       df/db2 = x/(b5*x + b6*x^2 + b7*x^3 + 1)
        //       df/db3 = x^2/(b5*x + b6*x^2 + b7*x^3 + 1)
        //       df/db4 = x^3/(b5*x + b6*x^2 + b7*x^3 + 1)
        //       df/db5 = -(x*(b1 + x*(b2 + x*(b3 + b4*x))))/(b5*x + b6*x^2 + b7*x^3 + 1)^2
        //       df/db6 = -(x^2*(b1 + x*(b2 + x*(b3 + b4*x))))/(b5*x + b6*x^2 + b7*x^3 + 1)^2
        //       df/db7 = -(x^3*(b1 + x*(b2 + x*(b3 + b4*x))))/(b5*x + b6*x^2 + b7*x^3 + 1)^2
        // best fitted parameters:
        //       b1 = 1.2881396800E+03 +/- 4.6647963344E+00
        //       b2 = 1.4910792535E+03 +/- 3.9571156086E+01
        //       b3 = 5.8323836877E+02 +/- 2.8698696102E+01
        //       b4 = 7.5416644291E+01 +/- 5.5675370270E+00
        //       b5 = 9.6629502864E-01 +/- 3.1333340687E-02
        //       b6 = 3.9797285797E-01 +/- 1.4984928198E-02
        //       b7 = 4.9727297349E-02 +/- 6.5842344623E-03
        private Vector<double> ThurberFunction(Vector<double> x, Vector<double> p)
        {
            var y = Vector<double>.Build.Dense(x.Count);
            for (int i = 0; i < x.Count; i++)
            {
                y[i] = (p[0] + p[1] * x[i] + p[2] * x[i] * x[i] + p[3] * x[i] * x[i] * x[i])
                    / (1 + p[4] * x[i] + p[5] * x[i] * x[i] + p[6] * x[i] * x[i] * x[i]);
            }
            return y;
        }
        private Matrix<double> ThurberPrime(Vector<double> x, Vector<double> p)
        {
            var prime = Matrix<double>.Build.Dense(x.Count, p.Count);
            for (int i = 0; i < x.Count; i++)
            {
                var xSq = x[i] * x[i];
                var xCb = xSq * x[i];
                var num = (p[0] + x[i] * (p[1] + x[i] * (p[2] + p[3] * x[i])));
                var den = (p[4] * x[i] + p[5] * xSq + p[6] * xCb + 1.0);
                var denSq = den * den;

                prime[i, 0] = 1.0 / den;
                prime[i, 1] = x[i] / den;
                prime[i, 2] = xSq / den;
                prime[i, 3] = xCb / den;
                prime[i, 4] = -(x[i] * num) / denSq;
                prime[i, 5] = -(xSq * num) / denSq;
                prime[i, 6] = -(xCb * num) / denSq;
            }
            return prime;
        }
        private Vector<double> Thurber_x = Vector<double>.Build.DenseOfArray(new[] {
            -3.067, -2.981, -2.921, -2.912, -2.84,
            -2.797, -2.702, -2.699, -2.633, -2.481,
            -2.363, -2.322, -1.501, -1.460, -1.274,
            -1.212, -1.100, -1.046, -0.915, -0.714,
            -0.566, -0.545, -0.400, -0.309, -0.109,
            -0.103, 0.01,   0.119,  0.377,  0.79,
            0.963,  1.006,  1.115,  1.572,  1.841,
            2.047,  2.2});
        private Vector<double> Thurber_y = Vector<double>.Build.DenseOfArray(new[] {
             80.574,    084.248,    087.264,    087.195,    089.076,
             089.608,   089.868,    090.101,    092.405,    095.854,
             100.696,   101.060,    401.672,    390.724,    567.534,
             635.316,   733.054,    759.087,    894.206,    990.785,
            1090.109,   1080.914,   1122.643,   1178.351,   1260.531,
            1273.514,   1288.339,   1327.543,   1353.863,   1414.509,
            1425.208,   1421.384,   1442.962,   1464.350,   1468.705,
            1447.894,   1457.628});
        private Vector<double> Thurber_p = Vector<double>.Build.DenseOfArray(new[] {
            1.2881396800E+03, 1.4910792535E+03, 5.8323836877E+02, 7.5416644291E+01, 9.6629502864E-01,
            3.9797285797E-01, 4.9727297349E-02 });

        [Test]
        public void LMDER_FindMinimum_Thurber_Unconstrained()
        {
            var obj = ObjectiveFunction.Jacobian(ThurberFunction, ThurberPrime, Thurber_x, Thurber_y);
            var solver = new LevenbergMarquardtMinimizer();
            var initialGuess = new DenseVector(new[] { 1000.0, 1000.0, 400.0, 40.0, 0.7, 0.3, 0.03 });

            var result = solver.FindMinimum(obj, initialGuess);

            for (int i = 0; i < result.MinimizingPoint.Count; i++)
            {
                AssertHelpers.AlmostEqualRelative(Thurber_p[i], result.MinimizingPoint[i], 6);
            }
        }

        [Test]
        public void LMDIF_FindMinimum_Thurber_Unconstrained()
        {
            var obj = ObjectiveFunction.Jacobian(ThurberFunction, Thurber_x, Thurber_y, accuracyOrder: 6);
            var solver = new LevenbergMarquardtMinimizer();
            var initialGuess = new DenseVector(new[] { 1000.0, 1000.0, 400.0, 40.0, 0.7, 0.3, 0.03 });

            var result = solver.FindMinimum(obj, initialGuess);

            for (int i = 0; i < result.MinimizingPoint.Count; i++)
            {
                AssertHelpers.AlmostEqualRelative(Thurber_p[i], result.MinimizingPoint[i], 6);
            }
        }
    }
}
